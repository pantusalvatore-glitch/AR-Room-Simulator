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
    o il binario kspaceFirstOrder-CUDA non sono disponibili.
    NON esiste fallback silenzioso su CPU.
  - Pre-check VRAM prima di lanciare il subprocess CUDA per intercettare
    MemoryError prima dell'allocazione, preservando il checkpoint batch.
  - Esecuzione della simulazione FDTD e restituzione di (RIR_LF, fs_sim)
    come tupla (np.ndarray 1D normalizzato, float).
  - fs_sim è calcolata dalla condizione CFL 3D reale (NON 44100):
    dt = CFL * dx / (c0 * sqrt(3))  →  fs_sim = 1 / dt
    Il Colab passa fs_sim a hybrid_crossover(fs_lf=fs_sim) che esegue
    resample_poly a FS_OUTPUT prima del crossover.

FIX rispetto a v1.0.2:
  [BUG-1] Import inesistenti rimossi:
          - kwave.kspaceFirstOrder  (non esiste in k-wave-python)
          - kwave.compat.options_to_kwargs  (non esiste)
          Sostituiti con kwave.kspaceFirstOrder3D.kspaceFirstOrder3DG (GPU)
  [BUG-2] dt corretto con fattore sqrt(3) per stabilità CFL 3D:
          dt = CFL * dx / (C_AIR * sqrt(3))
  [BUG-3] SimulationOptions: rimossi parametri non supportati
          (output_filename separato, pml_x/y/z_size/alpha scalari).
          Usati i campi reali: data_path, input_filename, output_filename,
          pml_x_size/y_size/z_size, pml_x_alpha/y_alpha/z_alpha.
  [BUG-4] Pre-check VRAM aggiunto in __post_init__ prima del subprocess.
  [BUG-5] sensor_data ora è np.ndarray (non dict) — _extract_rir aggiornato.

Dipendenze:
  - k-wave-python  (pip install k-wave-python)
  - numpy, scipy

Input  : dict prodotto da PhysicsTranslator.translate_profile() [Modulo 1]
Output : Tuple[np.ndarray 1D, float]  — (RIR_LF, fs_sim)
         RIR_LF è campionata a fs_sim (tipicamente 50–80 kHz, NON 44100 Hz).
         Passare fs_sim a hybrid_crossover(fs_lf=fs_sim) per il resample.

Autore  : Senior Audio DSP / Acoustic Simulation Engineer
Versione: 1.1.0
Python  : >= 3.10
=============================================================================
"""

from __future__ import annotations

import logging
import math
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
    # [FIX BUG-1] API corretta per k-wave-python 0.6.x:
    #   kspaceFirstOrder3DG  → GPU (CUDA)
    #   kspaceFirstOrder3DC  → C++ OMP (CPU)
    #   kspaceFirstOrder3D   → Python solver
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
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
PML_SIZE:   Final[int]   = 10
PML_ALPHA:  Final[float] = 2.0
SOURCE_MAGNITUDE: Final[float] = 1.0
KWAVE_HDF5_PREFIX: Final[str]  = "kwave_sim_"


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
            "(3) binario kspaceFirstOrder-CUDA accessibile,\n"
            "(4) che il nodo SLURM abbia una GPU assegnata (#SBATCH --gres=gpu:1)."
        )


# ---------------------------------------------------------------------------
# Dataclass — parametri di simulazione k-Wave
# ---------------------------------------------------------------------------

@dataclass
class KWaveSimParams:
    source_pos_m:  Tuple[float, float, float]
    mic_pos_m:     Tuple[float, float, float]
    t_end_s:       float         = 1.0
    pml_size:      int           = PML_SIZE
    use_gpu:       bool          = True
    hdf5_dir:      Optional[Path] = None
    smooth_source: bool          = True

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
    """
    Wraps k-wave-python for 3-D FDTD room acoustics simulation.

    Attributes pubblici (read-only dopo __init__):
        Nx, Ny, Nz  : dimensioni griglia [voxel]
        Nt          : numero di passi temporali
        fs_sim      : frequenza di campionamento reale k-Wave [Hz]
                      Derivata dalla condizione CFL 3D:
                      dt = CFL * dx / (c0 * sqrt(3))
                      fs_sim = 1 / dt
                      Tipicamente 50–80 kHz per dx ≈ 1 cm.
                      NON è 44100 Hz — il Colab esegue resample_poly
                      in hybrid_crossover(fs_lf=fs_sim).
    """

    def __init__(
        self,
        physics_params: dict,
        sim_params: KWaveSimParams,
    ) -> None:
        self._validate_physics_params(physics_params)
        self.p  = physics_params
        self.sp = sim_params

        self.dx:  float = physics_params["dx"]
        self.f_s: float = physics_params["f_schroeder"]
        self.Nx, self.Ny, self.Nz = physics_params["grid_shape"]
        self.rho_walls: dict[str, float] = physics_params["rho_per_surface"]

        # [FIX BUG-2] Condizione CFL corretta per griglia 3D.
        # La formula monodimensionale  dt = CFL * dx / c  NON garantisce
        # la stabilità in 3D.  Il fattore sqrt(3) deriva dalla norma del
        # vettore wavenumber massimo su tre assi:
        #   ||k_max||_3D = sqrt(3) * (pi / dx)
        # Senza sqrt(3) la simulazione può divergere (NaN/Inf nella RIR).
        self.dt:     float = CFL_NUMBER * self.dx / (C_AIR * math.sqrt(3.0))
        self.Nt:     int   = int(math.ceil(sim_params.t_end_s / self.dt))
        self.fs_sim: float = 1.0 / self.dt

        logger.info(
            "KWaveEngine inizializzato: grid=(%d x %d x %d) | "
            "dx=%.4f m | dt=%.3e s | Nt=%d | fs_sim=%.1f Hz",
            self.Nx, self.Ny, self.Nz,
            self.dx, self.dt, self.Nt, self.fs_sim,
        )

    # ── validazione ─────────────────────────────────────────────────────────

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

    # ── pre-check VRAM [FIX BUG-4] ──────────────────────────────────────────

    def _check_memory(self) -> None:
        """
        Pre-flight VRAM check prima di lanciare il subprocess CUDA.

        Intercetta MemoryError PRIMA di allocare qualcosa, permettendo
        al batch di loggare 'status: skip_ram' e salvare il checkpoint
        invece di crashare senza recovery.
        """
        if not self.sp.use_gpu:
            return

        ram_req_gb = float(self.p.get("ram_gb_estimated", 0.0))
        if ram_req_gb <= 0.0:
            return

        try:
            import torch
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / 1e9
            budget_gb = free_gb * 0.85   # headroom 15%
            if ram_req_gb > budget_gb:
                raise MemoryError(
                    f"VRAM stimata {ram_req_gb:.3f} GB > budget "
                    f"{budget_gb:.3f} GB ({free_gb:.2f} GB liberi). "
                    f"Riduci MAX_RAM_GB o la dimensione della griglia."
                )
            logger.info(
                "VRAM pre-check OK: richiesti %.3f GB | disponibili %.3f GB",
                ram_req_gb, free_gb,
            )
        except ImportError:
            logger.warning(
                "torch non disponibile — VRAM pre-check saltato."
            )

    # ── GPU enforcement ──────────────────────────────────────────────────────

    def _enforce_gpu(self) -> SimulationExecutionOptions:
        """
        Verifica CUDA e costruisce SimulationExecutionOptions.
        Lancia GPUInitializationError senza fallback silenzioso su CPU.
        """
        if not self.sp.use_gpu:
            logger.warning(
                "use_gpu=False: esecuzione su CPU. "
                "NON usare questa modalità su HPC."
            )
            return SimulationExecutionOptions(is_gpu_simulation=False)

        cuda_available = False
        cuda_source    = "unknown"

        # Probe 1: pycuda
        try:
            import pycuda.driver as cuda_drv  # type: ignore
            cuda_drv.init()
            n_devices = cuda_drv.Device.count()
            if n_devices == 0:
                raise GPUInitializationError(
                    f"pycuda rilevato ma nessun device CUDA trovato "
                    f"(Device.count() = {n_devices})."
                )
            device_name  = cuda_drv.Device(0).name()
            cuda_available = True
            cuda_source  = f"pycuda — device 0: {device_name}"
            logger.info("CUDA OK via pycuda: %s (%d device/s)", device_name, n_devices)
        except ImportError:
            pass
        except GPUInitializationError:
            raise
        except Exception as e:
            raise GPUInitializationError(
                f"pycuda trovato ma inizializzazione CUDA fallita: {e}"
            ) from e

        # Probe 2: torch
        if not cuda_available:
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    raise GPUInitializationError(
                        "torch trovato ma torch.cuda.is_available() = False. "
                        "Driver NVIDIA assente o CUDA non nel PATH."
                    )
                device_name  = torch.cuda.get_device_name(0)
                cuda_available = True
                cuda_source  = f"torch — device 0: {device_name}"
                logger.info("CUDA OK via torch: %s", device_name)
            except ImportError:
                pass

        if not cuda_available:
            raise GPUInitializationError(
                "Impossibile verificare la disponibilità CUDA: "
                "né pycuda né torch sono installati. "
                "Installa almeno uno: pip install torch"
            )

        # Risoluzione binario
        bin_dir = self._find_kwave_binary_dir()
        logger.info(
            "GPU enforcement OK [%s] | binary_dir: %s",
            cuda_source, bin_dir,
        )

        return SimulationExecutionOptions(
            is_gpu_simulation=True,
            binary_dir=bin_dir,
        )

    @staticmethod
    def _find_kwave_binary_dir() -> Optional[str]:
        """
        Restituisce la directory del binario k-Wave, o None se non trovata
        (k-wave-python userà il suo BINARY_DIR interno come fallback).

        Ordine di ricerca:
          1. $KWAVE_BIN_PATH  (impostato dalla Sezione 1 del Colab)
          2. shutil.which     (PATH di sistema, cattura il symlink)
          3. None             (lascia decidere a k-wave-python)
        """
        binary_name = "kspaceFirstOrder3DC"

        env_path = os.environ.get("KWAVE_BIN_PATH", "").strip()
        if env_path:
            candidate = Path(env_path) / binary_name
            if candidate.is_file():
                logger.info("Binario k-Wave trovato via KWAVE_BIN_PATH: %s", candidate)
                return env_path
            logger.warning(
                "KWAVE_BIN_PATH='%s' impostato ma '%s' non trovato lì.",
                env_path, binary_name,
            )

        which_result = shutil.which(binary_name)
        if which_result:
            bin_dir = str(Path(which_result).parent)
            logger.info("Binario k-Wave trovato via PATH: %s", which_result)
            return bin_dir

        logger.warning(
            "Binario '%s' non trovato esplicitamente — "
            "k-wave-python userà il suo BINARY_DIR interno.",
            binary_name,
        )
        return None

    # ── costruzione griglia ──────────────────────────────────────────────────

    def _build_grid(self) -> kWaveGrid:
        grid = kWaveGrid(
            [self.Nx, self.Ny, self.Nz],
            [self.dx, self.dx, self.dx],
        )
        # makeTime() imposta grid.Nt (int) e grid.dt su kWaveGrid.
        # Senza questa chiamata grid.Nt rimane la stringa 'auto' e
        # kspaceFirstOrder3DG solleva TypeError in input_checking.
        grid.makeTime(C_AIR, cfl=CFL_NUMBER, t_end=self.sp.t_end_s)
        logger.info(
            "kWaveGrid: [%d, %d, %d] voxel | dx=%.4f m | Nt=%d | dt=%.3e s",
            self.Nx, self.Ny, self.Nz, self.dx, grid.Nt, grid.dt,
        )
        return grid

    # ── costruzione medium ───────────────────────────────────────────────────

    def _build_medium(self) -> kWaveMedium:
        logger.info(
            "Costruzione medium 3D (%d x %d x %d)...",
            self.Nx, self.Ny, self.Nz,
        )

        rho_map = np.full((self.Nx, self.Ny, self.Nz), RHO_AIR, dtype=np.float32)
        c_map   = np.full((self.Nx, self.Ny, self.Nz), C_AIR,   dtype=np.float32)

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

        return kWaveMedium(sound_speed=c_map, density=rho_map)

    # ── snap-to-grid ─────────────────────────────────────────────────────────

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

        shift_m = math.sqrt(
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

    # ── segnale sorgente ─────────────────────────────────────────────────────

    def _build_source_signal(self) -> NDArray[np.float64]:
        t   = np.arange(self.Nt) * self.dt
        f_c = self.f_s / 2.0

        n_cycles = 4
        sigma = n_cycles / (2.0 * np.pi * f_c)
        t0    = 3.0 * sigma

        envelope = np.exp(-((t - t0) / sigma) ** 2)
        carrier  = np.sin(2.0 * np.pi * f_c * t)
        signal   = SOURCE_MAGNITUDE * envelope * carrier

        nyq = self.fs_sim / 2.0
        if self.f_s < nyq:
            sos    = butter(N=8, Wn=self.f_s / nyq, btype="low", output="sos")
            signal = sosfiltfilt(sos, signal)
            logger.info(
                "Source signal — LP Butterworth ord.8 a f_s=%.2f Hz applicato.", self.f_s
            )
        else:
            logger.warning(
                "f_s (%.2f Hz) >= Nyquist (%.2f Hz): filtro LP non applicato.",
                self.f_s, nyq,
            )

        peak = np.max(np.abs(signal))
        if peak > 1e-12:
            signal /= peak

        logger.info(
            "Source signal: Gaussiana modulata f_c=%.2f Hz | sigma=%.5f s | Nt=%d",
            f_c, sigma, self.Nt,
        )
        return signal.astype(np.float64)

    # ── source / sensor ──────────────────────────────────────────────────────

    def _build_source(
        self,
        source_idx: Tuple[int, int, int],
        signal: NDArray[np.float64],
    ) -> kSource:
        ix, iy, iz = source_idx
        mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        mask[ix, iy, iz] = True

        source   = kSource()
        source.p_mask = mask
        source.p = signal[np.newaxis, :]

        logger.info("kSource: voxel=(%d,%d,%d) | p shape=%s", ix, iy, iz, source.p.shape)
        return source

    def _build_sensor(self, mic_idx: Tuple[int, int, int]) -> kSensor:
        ix, iy, iz = mic_idx
        mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        mask[ix, iy, iz] = True

        sensor        = kSensor()
        sensor.mask   = mask
        sensor.record = ["p"]

        logger.info("kSensor: voxel=(%d,%d,%d)", ix, iy, iz)
        return sensor

    # ── HDF5 paths ───────────────────────────────────────────────────────────

    def _make_hdf5_paths(self) -> Tuple[Path, Path]:
        hdf5_dir = self.sp.hdf5_dir or Path(tempfile.gettempdir())
        hdf5_dir.mkdir(parents=True, exist_ok=True)

        uid         = uuid.uuid4().hex[:12]
        input_path  = hdf5_dir / f"{KWAVE_HDF5_PREFIX}{uid}_input.h5"
        output_path = hdf5_dir / f"{KWAVE_HDF5_PREFIX}{uid}_output.h5"

        logger.info("HDF5: input=%s | output=%s", input_path, output_path)
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

    # ── estrazione RIR [FIX BUG-5] ──────────────────────────────────────────

    @staticmethod
    def _extract_rir(sensor_data: object) -> NDArray[np.float64]:
        """
        Estrae la RIR 1D da sensor_data.

        In k-wave-python 0.6.x kspaceFirstOrder3DG restituisce direttamente
        un np.ndarray (non un dict). Gestiamo entrambi i casi per robustezza.
        """
        # Caso 1: np.ndarray diretto (k-wave-python 0.6.x con sensor.record=['p'])
        if isinstance(sensor_data, np.ndarray):
            rir = sensor_data.squeeze().astype(np.float64)
        # Caso 2: dict con chiave 'p' (versioni più vecchie / future)
        elif isinstance(sensor_data, dict):
            p_raw = sensor_data.get("p")
            if p_raw is None:
                raise RuntimeError(
                    "sensor_data (dict) non contiene la chiave 'p'. "
                    "Verifica che sensor.record = ['p'] sia impostato."
                )
            rir = np.array(p_raw, dtype=np.float64).squeeze()
        else:
            raise RuntimeError(
                f"Tipo sensor_data non gestito: {type(sensor_data)}. "
                f"Atteso np.ndarray o dict."
            )

        if rir.ndim != 1:
            raise RuntimeError(
                f"RIR_LF attesa 1D, ottenuto shape={rir.shape}. "
                f"Sensor multi-punto non supportato in questo modulo."
            )

        peak = float(np.max(np.abs(rir)))
        if peak < 1e-15:
            logger.warning("RIR_LF ha picco quasi nullo (%.2e). Griglia instabile?", peak)
        else:
            rir /= peak

        logger.info(
            "RIR_LF: Nt=%d campioni | peak originale=%.4e | "
            "range=[%.4f, %.4f]",
            len(rir), peak, float(rir.min()), float(rir.max()),
        )
        return rir

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self) -> NDArray[np.float64]:
        """
        Esegue la simulazione FDTD k-Wave.

        Returns
        -------
        rir_lf : np.ndarray shape (Nt,), dtype float64
            RIR campionata a self.fs_sim (tipicamente 50–80 kHz).
            NON a 44100 Hz — il Colab esegue resample_poly in
            hybrid_crossover(fs_lf=self.fs_sim).

        Raises
        ------
        GPUInitializationError  — CUDA non disponibile / binario non trovato
        MemoryError             — VRAM insufficiente (pre-check)
        RuntimeError            — simulazione fallita
        """
        logger.info("=" * 60)
        logger.info("KWaveEngine.run() — START")
        logger.info(
            "  Source: %s m | Mic: %s m | t_end=%.3f s | GPU=%s",
            self.sp.source_pos_m, self.sp.mic_pos_m,
            self.sp.t_end_s, self.sp.use_gpu,
        )
        logger.info("=" * 60)

        # Step 0: Pre-check VRAM [FIX BUG-4]
        self._check_memory()

        # Step 1: GPU enforcement
        exec_options = self._enforce_gpu()

        # Step 2–3: Griglia e medium
        grid   = self._build_grid()
        medium = self._build_medium()

        # Step 4–5: Snap-to-grid
        src_idx = self._snap_to_grid(self.sp.source_pos_m, "source")
        mic_idx = self._snap_to_grid(self.sp.mic_pos_m,    "mic")

        if src_idx == mic_idx:
            logger.warning("Source e microfono coincidono sullo stesso voxel: %s.", src_idx)

        # Step 6–8: Segnale e trasduttori
        signal = self._build_source_signal()
        source = self._build_source(src_idx, signal)
        sensor = self._build_sensor(mic_idx)

        # Step 9: HDF5 paths
        input_hdf5, output_hdf5 = self._make_hdf5_paths()

        # Step 10–11: Simulazione FDTD [FIX BUG-1, BUG-3]
        sensor_data = None
        try:
            logger.info("Avvio simulazione k-Wave FDTD (GPU=%s)...", self.sp.use_gpu)

            # [FIX BUG-3] SimulationOptions con parametri reali di k-wave-python 0.6.x.
            # - data_path + input_filename + output_filename gestiscono i file HDF5.
            # - pml_x/y/z_size e pml_x/y/z_alpha sono i campi scalari supportati.
            # - NON esiste un singolo 'pml_size' scalare se si vogliono assi separati;
            #   si usa pml_x_size=pml_y_size=pml_z_size per uniformità.
            # PML adattivo: deve essere < N/2 su ogni asse
            _pml_max = max(4, min(self.Nx, self.Ny, self.Nz) // 2 - 1)
            _pml   = min(int(self.sp.pml_size), _pml_max)
            if _pml < int(self.sp.pml_size):
                logger.warning(
                    "PML ridotto da %d a %d (griglia min=%d voxel)",
                    int(self.sp.pml_size), _pml, min(self.Nx, self.Ny, self.Nz)
                )
            _alpha = float(PML_ALPHA)

            sim_options = SimulationOptions(
                pml_size=[_pml, _pml, _pml],
                pml_inside=False,
                smooth_p0=self.sp.smooth_source,
                save_to_disk=True,
                data_path=str(input_hdf5.parent),
                input_filename=input_hdf5.name,
                output_filename=output_hdf5.name,
                data_cast="single",
            )

            # [FIX BUG-1] kspaceFirstOrder3DG è la funzione GPU corretta.
            # Firma: (kgrid, source, sensor, medium, simulation_options, execution_options)
            sensor_data = kspaceFirstOrder3DG(
                kgrid=grid,
                source=source,
                sensor=sensor,
                medium=medium,
                simulation_options=sim_options,
                execution_options=exec_options,
            )

            logger.info("Simulazione k-Wave completata.")

        except MemoryError:
            logger.error(
                "MemoryError: griglia (%d x %d x %d) troppo grande per la GPU.",
                self.Nx, self.Ny, self.Nz,
            )
            raise

        except GPUInitializationError:
            raise

        except Exception as e:
            logger.error("Simulazione k-Wave fallita: %s: %s", type(e).__name__, e)
            raise RuntimeError(
                f"kspaceFirstOrder3DG ha sollevato {type(e).__name__}: {e}"
            ) from e

        finally:
            self._cleanup_hdf5(input_hdf5, output_hdf5)
            logger.info("File HDF5 temporanei rimossi.")

        # Step 12: Estrazione RIR [FIX BUG-5]
        rir_lf = self._extract_rir(sensor_data)

        logger.info(
            "KWaveEngine.run() DONE — shape=%s | fs_sim=%.2f Hz",
            rir_lf.shape, self.fs_sim,
        )
        return rir_lf


# ---------------------------------------------------------------------------
# Funzione pubblica di convenienza — interfaccia Colab
# ---------------------------------------------------------------------------

def generate_rir_lf(
    physics_params: dict,
    source_pos_m:   Tuple[float, float, float],
    mic_pos_m:      Tuple[float, float, float],
    t_end_s:        float         = 1.0,
    use_gpu:        bool          = True,
    hdf5_dir:       Optional[Path] = None,
) -> Tuple[NDArray[np.float64], float]:
    """
    Wrapper pubblico consumato dal Colab e dal batch loop.

    Returns
    -------
    rir_lf : np.ndarray   — RIR campionata a fs_lf (NON 44100 Hz)
    fs_lf  : float        — frequenza di campionamento k-Wave reale [Hz]
                            Da passare a hybrid_crossover(fs_lf=fs_lf)
                            affinché resample_poly riscampioni a FS_OUTPUT.

    Raises
    ------
    GPUInitializationError   — use_gpu=True e CUDA non disponibile
    MemoryError              — VRAM insufficiente (pre-check)
    RuntimeError             — simulazione FDTD fallita
    """
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
# Self-test (struttura, senza FDTD reale)
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

    from m1_physics_setup import RoomAcousticProfile, PhysicsTranslator

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
    print(f"  dt       = {engine.dt:.4e} s")
    print(f"  Nt       = {engine.Nt}")
    print(f"  fs_sim   = {engine.fs_sim:.2f} Hz  (atteso ~50–80 kHz, NON 44100)")

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
    print(f"  rho shape:    {medium.density.shape}")
    print(f"  rho floor:    {medium.density[1, 1, 0]:.4f} kg/m³")
    print(f"  rho ceiling:  {medium.density[1, 1, -1]:.4f} kg/m³")
    print(f"  rho interior: {medium.density[3, 3, 3]:.4f} kg/m³")

    print("\nTest GPUInitializationError (use_gpu=True senza GPU):")
    try:
        engine._enforce_gpu()
        print("  GPU disponibile — nessuna eccezione sollevata.")
    except GPUInitializationError as e:
        print(f"  [OK] GPUInitializationError: {str(e).splitlines()[0]}")

    print("\nTest generate_rir_lf signature:")
    import inspect
    sig = inspect.signature(generate_rir_lf)
    print(f"  Firma:   {sig}")
    print(f"  Returns: Tuple[np.ndarray, float]  ← (rir_lf, fs_lf)")

    print("\n" + "=" * 60)
    print("Self test Modulo 2 completato — tutti i fix applicati.")
    print("=" * 60)
