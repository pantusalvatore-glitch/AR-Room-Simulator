"""
=============================================================================
MODULO 2 — k-Wave FDTD Engine (Low-Frequency RIR)
=============================================================================
Simulatore Acustico Ibrido XR — HPC-Ready Pipeline

Responsabilità:
  - Ricezione del dizionario fisico prodotto dal Modulo 1.
  - Costruzione della griglia k-Wave (kWaveGrid) con passo dx.
  - "Pittura" del medium 3D: assegnazione di rho e c a ogni voxel,
    con boundary walls che rispecchiano le impedenze acustiche reali.
  - Posizionamento snap-to-grid di sorgente e microfono.
  - Generazione di un segnale sorgente impulsivo band-limited (Gaussiana
    modulata + filtro Butterworth passa-basso a f_s) per evitare
    aliasing spaziale sulla griglia FDTD.
  - Enforcement GPU esplicito: lancia GPUInitializationError se CUDA
    o il binario kspaceFirstOrder3DC non sono disponibili.
    NON esiste fallback silenzioso su CPU.
  - Esecuzione della simulazione FDTD e restituzione di RIR_LF come
    array numpy 1D normalizzato.
  - Gestione sicura dei file temporanei HDF5 (cleanup garantito anche
    in caso di eccezione).

Dipendenze:
  - k-wave-python  (pip install k-wave-python)
  - numpy, scipy

Input  : dict prodotto da PhysicsTranslator.translate_profile() [Modulo 1]
Output : np.ndarray 1D  — RIR_LF (float64, campionata a fs_sim)

Autore  : Senior Audio DSP / Acoustic Simulation Engineer
Versione: 1.0.2
Python  : >= 3.10
=============================================================================
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# k-wave-python imports — ImportError esplicito e leggibile
# ---------------------------------------------------------------------------
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    from kwave.compat import options_to_kwargs
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
except ImportError as _kwave_err:
    raise ImportError(
        "k-wave-python non trovato. Installalo con:\n"
        "    pip install k-wave-python\n"
        f"Errore originale: {_kwave_err}"
    ) from _kwave_err

# ---------------------------------------------------------------------------
# Import Modulo 1
# ---------------------------------------------------------------------------
from m1_physics_setup import (
    C_AIR,
    RHO_AIR,
    SURFACE_KEYS,
    RoomAcousticProfile,
    PhysicsTranslator,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
CFL_NUMBER: Final[float] = 0.3
PML_SIZE: Final[int] = 10
PML_ALPHA: Final[float] = 2.0
SOURCE_MAGNITUDE: Final[float] = 1.0
KWAVE_HDF5_PREFIX: Final[str] = "kwave_sim_"


# ---------------------------------------------------------------------------
# Eccezione custom GPU
# ---------------------------------------------------------------------------

class GPUInitializationError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"[GPU INIT FAILED] {reason}\n"
            "Il sistema richiede CUDA per l'esecuzione k-Wave su HPC.\n"
            "Verifica: (1) driver NVIDIA installati, (2) CUDA toolkit nel PATH,\n"
            "(3) binario kspaceFirstOrder3DC compilato e accessibile,\n"
            "(4) che il nodo SLURM abbia una GPU assegnata (#SBATCH --gres=gpu:1)."
        )


# ---------------------------------------------------------------------------
# Dataclass — parametri di simulazione k-Wave
# ---------------------------------------------------------------------------

@dataclass
class KWaveSimParams:
    source_pos_m:   Tuple[float, float, float]
    mic_pos_m:      Tuple[float, float, float]
    fs_simulation:  float = 0.0
    t_end_s:        float = 1.0
    pml_size:       int   = PML_SIZE
    use_gpu:        bool  = True
    hdf5_dir:       Optional[Path] = None
    smooth_source:  bool  = True

    def __post_init__(self) -> None:
        for label, pos in [("source_pos_m", self.source_pos_m),
                            ("mic_pos_m",    self.mic_pos_m)]:
            if len(pos) != 3:
                raise ValueError(
                    f"KWaveSimParams.{label} deve essere una tupla (x, y, z), "
                    f"ricevuto {pos}."
                )
            if any(v < 0 for v in pos):
                raise ValueError(
                    f"KWaveSimParams.{label} = {pos}: le coordinate devono "
                    f"essere >= 0."
                )
        if self.t_end_s <= 0.0:
            raise ValueError(f"t_end_s deve essere > 0, ricevuto {self.t_end_s}.")
        if self.pml_size < 4:
            raise ValueError(
                f"pml_size={self.pml_size} troppo piccolo (min 4 voxel) "
                f"per PML efficace."
            )


# ---------------------------------------------------------------------------
# Engine principale
# ---------------------------------------------------------------------------

class KWaveEngine:

    def __init__(
        self,
        physics_params: dict,
        sim_params: KWaveSimParams,
    ) -> None:
        self._validate_physics_params(physics_params)
        self.p  = physics_params
        self.sp = sim_params

        self.dx:  float               = physics_params["dx"]
        self.f_s: float               = physics_params["f_schroeder"]
        self.Nx, self.Ny, self.Nz     = physics_params["grid_shape"]
        self.rho_walls: dict[str, float] = physics_params["rho_per_surface"]

        self.dt: float  = CFL_NUMBER * self.dx / C_AIR
        self.Nt: int    = int(np.ceil(sim_params.t_end_s / self.dt))
        self.fs_sim: float = 1.0 / self.dt

        logger.info(
            "KWaveEngine inizializzato: grid=(%d x %d x %d) | "
            "dx=%.4f m | dt=%.3e s | Nt=%d | fs_sim=%.1f Hz",
            self.Nx, self.Ny, self.Nz,
            self.dx, self.dt, self.Nt, self.fs_sim,
        )

    @staticmethod
    def _validate_physics_params(params: dict) -> None:
        required_keys = {
            "f_schroeder", "dx", "grid_shape",
            "rho_per_surface", "volume", "t60",
        }
        missing = required_keys - set(params.keys())
        if missing:
            raise KeyError(
                f"physics_params manca delle chiavi: {missing}. "
                f"Assicurati di passare l'output di "
                f"PhysicsTranslator.translate_profile()."
            )
        if params["dx"] <= 0:
            raise ValueError(f"physics_params['dx'] = {params['dx']} deve essere > 0.")
        if len(params["grid_shape"]) != 3:
            raise ValueError("physics_params['grid_shape'] deve essere (Nx, Ny, Nz).")

    def _enforce_gpu(self) -> SimulationExecutionOptions:
        if not self.sp.use_gpu:
            logger.warning(
                "use_gpu=False: esecuzione su CPU. "
                "NON usare questa modalità su HPC."
            )
            return SimulationExecutionOptions(is_gpu_simulation=False)

        cuda_available = False
        cuda_source = "unknown"

        try:
            import pycuda.driver as cuda_drv  # type: ignore
            cuda_drv.init()
            n_devices = cuda_drv.Device.count()
            if n_devices == 0:
                raise GPUInitializationError(
                    "pycuda rilevato ma nessun device CUDA trovato "
                    f"(Device.count() = {n_devices})."
                )
            device_name = cuda_drv.Device(0).name()
            cuda_available = True
            cuda_source = f"pycuda — device 0: {device_name}"
            logger.info("CUDA OK via pycuda: %s (%d device/s)", device_name, n_devices)
        except ImportError:
            pass
        except Exception as e:
            raise GPUInitializationError(
                f"pycuda trovato ma inizializzazione CUDA fallita: {e}"
            ) from e

        if not cuda_available:
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    raise GPUInitializationError(
                        "torch trovato ma torch.cuda.is_available() = False. "
                        "Driver NVIDIA assente o CUDA non nel PATH."
                    )
                device_name = torch.cuda.get_device_name(0)
                cuda_available = True
                cuda_source = f"torch — device 0: {device_name}"
                logger.info("CUDA OK via torch: %s", device_name)
            except ImportError:
                pass

        if not cuda_available:
            raise GPUInitializationError(
                "Impossibile verificare la disponibilità CUDA: "
                "né pycuda né torch sono installati. "
                "Installa almeno uno dei due come proxy per il driver NVIDIA:\n"
                "    pip install pycuda   oppure   pip install torch"
            )

        kwave_bin = self._find_kwave_binary()
        logger.info(
            "GPU enforcement OK [%s] | binario k-Wave: %s",
            cuda_source, kwave_bin,
        )

        env_path = os.environ.get("KWAVE_BIN_PATH", "")
        exec_options = SimulationExecutionOptions(
            is_gpu_simulation=True,
            binary_dir=env_path if env_path else None,
        )
        return exec_options

    @staticmethod
    def _find_kwave_binary() -> str:
        binary_name = "kspaceFirstOrder3DC"

        env_path = os.environ.get("KWAVE_BIN_PATH", "")
        if env_path:
            candidate = Path(env_path) / binary_name
            if candidate.is_file():
                return str(candidate)
            logger.warning(
                "KWAVE_BIN_PATH='%s' impostato ma binario non trovato in quella path.",
                env_path,
            )

        which_result = shutil.which(binary_name)
        if which_result:
            return which_result

        local_candidate = Path.cwd() / binary_name
        if local_candidate.is_file():
            return str(local_candidate)

        raise GPUInitializationError(
            f"Binario '{binary_name}' non trovato.\n"
            "Soluzioni:\n"
            "  (a) Imposta la variabile d'ambiente KWAVE_BIN_PATH=/path/to/bin/\n"
            "  (b) Aggiungi la directory del binario al PATH di sistema\n"
            "  (c) Copia il binario nella directory di lavoro corrente\n"
            "Il binario si compila dal sorgente C++ di k-Wave "
            "(http://www.k-wave.org/download.php)."
        )

    def _build_grid(self) -> kWaveGrid:
        grid = kWaveGrid(
            [self.Nx, self.Ny, self.Nz],
            [self.dx, self.dx, self.dx],
        )
        logger.info(
            "kWaveGrid costruita: [%d, %d, %d] voxel | dx=%.4f m",
            self.Nx, self.Ny, self.Nz, self.dx,
        )
        return grid

    def _build_medium(self) -> kWaveMedium:
        logger.info("Costruzione medium 3D (%d x %d x %d)...", self.Nx, self.Ny, self.Nz)

        rho_map = np.full((self.Nx, self.Ny, self.Nz), RHO_AIR,  dtype=np.float32)
        c_map   = np.full((self.Nx, self.Ny, self.Nz), C_AIR,    dtype=np.float32)

        rho_map[0,  :, :] = self.rho_walls["wall_front"]
        rho_map[-1, :, :] = self.rho_walls["wall_back"]
        rho_map[:, 0,  :] = self.rho_walls["wall_left"]
        rho_map[:, -1, :] = self.rho_walls["wall_right"]
        rho_map[:, :, 0 ] = self.rho_walls["floor"]
        rho_map[:, :, -1] = self.rho_walls["ceiling"]

        n_boundary = (
            2 * self.Ny * self.Nz +
            2 * self.Nx * self.Nz +
            2 * self.Nx * self.Ny
        )
        n_interior = self.Nx * self.Ny * self.Nz - n_boundary
        logger.info(
            "Medium painted: %d voxel interni (aria) | %d voxel boundary",
            n_interior, n_boundary,
        )

        medium = kWaveMedium(
            sound_speed=c_map,
            density=rho_map,
        )
        return medium

    def _snap_to_grid(
        self,
        pos_m: Tuple[float, float, float],
        label: str,
    ) -> Tuple[int, int, int]:
        x_m, y_m, z_m = pos_m
        L = self.Nx * self.dx
        W = self.Ny * self.dx
        H = self.Nz * self.dx

        if x_m > L or y_m > W or z_m > H:
            raise ValueError(
                f"Posizione {label} {pos_m} m fuori dalla stanza "
                f"({L:.2f} x {W:.2f} x {H:.2f} m)."
            )

        ix = int(np.clip(round(x_m / self.dx), 1, self.Nx - 2))
        iy = int(np.clip(round(y_m / self.dx), 1, self.Ny - 2))
        iz = int(np.clip(round(z_m / self.dx), 1, self.Nz - 2))

        shift_m = np.sqrt(
            (ix * self.dx - x_m) ** 2 +
            (iy * self.dx - y_m) ** 2 +
            (iz * self.dx - z_m) ** 2
        )
        if shift_m > 0.5 * self.dx:
            logger.warning(
                "Snap-to-grid '%s': shift %.4f m (> dx/2 = %.4f m).",
                label, shift_m, 0.5 * self.dx,
            )
        else:
            logger.info(
                "Snap-to-grid '%s': (%.3f, %.3f, %.3f) m -> voxel (%d, %d, %d) "
                "| shift=%.4f m",
                label, x_m, y_m, z_m, ix, iy, iz, shift_m,
            )

        return ix, iy, iz

    def _build_source_signal(self) -> NDArray[np.float64]:
        t = np.arange(self.Nt) * self.dt
        f_c = self.f_s / 2.0

        n_cycles = 4
        sigma = n_cycles / (2.0 * np.pi * f_c)
        t0    = 3.0 * sigma

        envelope = np.exp(-((t - t0) / sigma) ** 2)
        carrier  = np.sin(2.0 * np.pi * f_c * t)
        signal   = SOURCE_MAGNITUDE * envelope * carrier

        logger.info(
            "Source signal — Gaussiana modulata: f_c=%.2f Hz, sigma=%.5f s, "
            "t0=%.5f s, Nt=%d",
            f_c, sigma, t0, self.Nt,
        )

        nyq = self.fs_sim / 2.0
        if self.f_s < nyq:
            sos = butter(N=8, Wn=self.f_s / nyq, btype="low", output="sos")
            signal = sosfiltfilt(sos, signal)
            logger.info("Source signal — LP Butterworth ord.8 a f_s=%.2f Hz applicato.", self.f_s)
        else:
            logger.warning("f_s (%.2f Hz) >= Nyquist (%.2f Hz): filtro LP non applicato.", self.f_s, nyq)

        peak = np.max(np.abs(signal))
        if peak > 1e-12:
            signal /= peak

        return signal.astype(np.float64)

    def _build_source(
        self,
        source_idx: Tuple[int, int, int],
        signal: NDArray[np.float64],
    ) -> kSource:
        ix, iy, iz = source_idx

        source_mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        source_mask[ix, iy, iz] = True

        source = kSource()
        source.p_mask = source_mask
        source.p = signal[np.newaxis, :]

        logger.info("kSource: voxel=(%d, %d, %d) | segnale shape=%s", ix, iy, iz, source.p.shape)
        return source

    def _build_sensor(self, mic_idx: Tuple[int, int, int]) -> kSensor:
        ix, iy, iz = mic_idx

        sensor_mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        sensor_mask[ix, iy, iz] = True

        sensor = kSensor()
        sensor.mask = sensor_mask
        sensor.record = ["p"]

        logger.info("kSensor: voxel=(%d, %d, %d)", ix, iy, iz)
        return sensor

    def _make_hdf5_paths(self) -> Tuple[Path, Path]:
        hdf5_dir = self.sp.hdf5_dir or Path(tempfile.gettempdir())
        hdf5_dir.mkdir(parents=True, exist_ok=True)

        uid = uuid.uuid4().hex[:12]
        input_path  = hdf5_dir / f"{KWAVE_HDF5_PREFIX}{uid}_input.h5"
        output_path = hdf5_dir / f"{KWAVE_HDF5_PREFIX}{uid}_output.h5"

        logger.info("HDF5 temporanei: input=%s | output=%s", input_path, output_path)
        return input_path, output_path

    @staticmethod
    def _cleanup_hdf5(*paths: Path) -> None:
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
                    logger.debug("HDF5 eliminato: %s", p)
            except OSError as e:
                logger.warning("Impossibile eliminare HDF5 %s: %s", p, e)

    @staticmethod
    def _extract_rir(sensor_data: dict) -> NDArray[np.float64]:
        p_raw = sensor_data.get("p", None)
        if p_raw is None:
            raise RuntimeError(
                "sensor_data non contiene la chiave 'p'. "
                "Verifica che sensor.record = ['p'] sia impostato correttamente."
            )

        rir = np.array(p_raw, dtype=np.float64).squeeze()

        if rir.ndim != 1:
            raise RuntimeError(
                f"RIR_LF attesa 1D, ottenuto shape={rir.shape}. "
                f"Sensor multi-punto non supportato in questo modulo."
            )

        peak = np.max(np.abs(rir))
        if peak < 1e-15:
            logger.warning("RIR_LF ha picco quasi nullo (%.2e).", peak)
        else:
            rir /= peak

        logger.info(
            "RIR_LF estratta: Nt=%d campioni | peak originale=%.4e | "
            "range normalizzato=[%.4f, %.4f]",
            len(rir), peak, float(rir.min()), float(rir.max()),
        )
        return rir

    def run(self) -> NDArray[np.float64]:
        logger.info("=" * 60)
        logger.info("KWaveEngine.run() — START")
        logger.info(
            "  Source: %s m | Mic: %s m | t_end=%.3f s | GPU=%s",
            self.sp.source_pos_m, self.sp.mic_pos_m,
            self.sp.t_end_s, self.sp.use_gpu,
        )
        logger.info("=" * 60)

        # Step 1: GPU enforcement
        exec_options = self._enforce_gpu()

        # Step 2-3: Griglia e medium
        grid   = self._build_grid()
        medium = self._build_medium()

        # Step 4-5: Snap-to-grid
        src_idx = self._snap_to_grid(self.sp.source_pos_m, "source")
        mic_idx = self._snap_to_grid(self.sp.mic_pos_m,    "mic")

        if src_idx == mic_idx:
            logger.warning(
                "Source e microfono cadono nello stesso voxel (%s).", src_idx,
            )

        # Step 6-8: Segnale e trasduttori
        signal = self._build_source_signal()
        source = self._build_source(src_idx, signal)
        sensor = self._build_sensor(mic_idx)

        # Step 9: Path HDF5
        input_hdf5, output_hdf5 = self._make_hdf5_paths()

        # Step 10-11: Simulazione FDTD
        sensor_data: Optional[dict] = None
        try:
            logger.info("Avvio simulazione k-Wave FDTD (GPU)...")

            _pml = int(self.sp.pml_size)
            sim_options = SimulationOptions(
                pml_x_size=_pml,
                pml_y_size=_pml,
                pml_z_size=_pml,
                pml_x_alpha=float(PML_ALPHA),
                pml_y_alpha=float(PML_ALPHA),
                pml_z_alpha=float(PML_ALPHA),
                smooth_p0=self.sp.smooth_source,
                save_to_disk=True,
                input_filename=str(input_hdf5),
                output_filename=str(output_hdf5),
                data_cast="single",
            )

            # FIX: nuova API kspaceFirstOrder + options_to_kwargs
            kwargs = options_to_kwargs(sim_options, exec_options)
            sensor_data = kspaceFirstOrder(
                kgrid=grid,
                source=source,
                sensor=sensor,
                medium=medium,
                **kwargs,
            )

            logger.info("Simulazione k-Wave completata.")

        except MemoryError:
            logger.error(
                "MemoryError: griglia (%d x %d x %d) troppo grande per la GPU.",
                self.Nx, self.Ny, self.Nz,
            )
            raise

        except Exception as e:
            logger.error("Simulazione k-Wave fallita: %s: %s", type(e).__name__, e)
            raise RuntimeError(
                f"kspaceFirstOrder ha sollevato {type(e).__name__}: {e}"
            ) from e

        finally:
            self._cleanup_hdf5(input_hdf5, output_hdf5)
            logger.info("File HDF5 temporanei rimossi.")

        # Step 12: Estrazione RIR
        rir_lf = self._extract_rir(sensor_data)

        logger.info(
            "KWaveEngine.run() DONE — RIR_LF shape=%s | fs_sim=%.2f Hz",
            rir_lf.shape, self.fs_sim,
        )
        return rir_lf


# ---------------------------------------------------------------------------
# Funzione di convenienza
# ---------------------------------------------------------------------------

def generate_rir_lf(
    physics_params: dict,
    source_pos_m:   Tuple[float, float, float],
    mic_pos_m:      Tuple[float, float, float],
    t_end_s:        float = 1.0,
    use_gpu:        bool  = True,
    hdf5_dir:       Optional[Path] = None,
) -> Tuple[NDArray[np.float64], float]:
    sim_params = KWaveSimParams(
        source_pos_m=source_pos_m,
        mic_pos_m=mic_pos_m,
        t_end_s=t_end_s,
        use_gpu=use_gpu,
        hdf5_dir=hdf5_dir,
    )
    engine = KWaveEngine(physics_params=physics_params, sim_params=sim_params)
    rir_lf = engine.run()
    return rir_lf, engine.fs_sim


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("MODULO 2 — Self Test (struttura, senza FDTD reale)")
    print("=" * 60)

    from m1_physics_setup import RoomAcousticProfile, PhysicsTranslator, SURFACE_KEYS

    room = RoomAcousticProfile(
        length=6.0, width=4.0, height=3.0,
        alpha_per_surface={
            "floor":      {125: 0.02, 250: 0.03},
            "ceiling":    {125: 0.15, 250: 0.20},
            "wall_front": {125: 0.05, 250: 0.07},
            "wall_back":  {125: 0.40, 250: 0.45},
            "wall_left":  {125: 0.05, 250: 0.07},
            "wall_right": {125: 0.05, 250: 0.07},
        },
    )

    physics = PhysicsTranslator.translate_profile(
        profile=room, target_freq_hz=125, max_ram_gb=16.0,
    )

    print(f"\nFisica Modulo 1:")
    print(f"  f_schroeder = {physics['f_schroeder']:.2f} Hz")
    print(f"  dx          = {physics['dx']*100:.3f} cm")
    print(f"  grid_shape  = {physics['grid_shape']}")

    sim_params = KWaveSimParams(
        source_pos_m=(1.0, 1.0, 1.0),
        mic_pos_m=(4.5, 3.0, 1.5),
        t_end_s=0.5,
        use_gpu=True,
    )

    engine = KWaveEngine(physics_params=physics, sim_params=sim_params)

    print(f"\nKWaveEngine parametri calcolati:")
    print(f"  dt     = {engine.dt:.4e} s")
    print(f"  Nt     = {engine.Nt}")
    print(f"  fs_sim = {engine.fs_sim:.2f} Hz")

    print("\nTest snap-to-grid:")
    src_idx = engine._snap_to_grid((1.0, 1.0, 1.0), "source")
    mic_idx = engine._snap_to_grid((4.5, 3.0, 1.5), "mic")
    print(f"  Source: (1.0, 1.0, 1.0) m -> voxel {src_idx}")
    print(f"  Mic:    (4.5, 3.0, 1.5) m -> voxel {mic_idx}")

    print("\nTest source signal:")
    signal = engine._build_source_signal()
    print(f"  Shape: {signal.shape} | Peak: {float(np.max(np.abs(signal))):.6f}")

    print("\nTest medium painting:")
    medium = engine._build_medium()
    print(f"  rho shape: {medium.density.shape}")
    print(f"  rho floor:    {medium.density[1, 1, 0]:.4f} kg/m3")
    print(f"  rho ceiling:  {medium.density[1, 1, -1]:.4f} kg/m3")
    print(f"  rho interior: {medium.density[3, 3, 3]:.4f} kg/m3")

    print("\nTest GPUInitializationError:")
    try:
        engine._enforce_gpu()
        print("  GPU disponibile — nessuna eccezione sollevata.")
    except GPUInitializationError as e:
        print(f"  [OK] GPUInitializationError: {str(e).splitlines()[0]}")

    print("\n" + "=" * 60)
    print("Self test Modulo 2 completato.")
    print("=" * 60)
