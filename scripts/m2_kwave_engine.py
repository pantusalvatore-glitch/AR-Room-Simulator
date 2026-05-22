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
Versione: 1.0.1
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
    # FIX: import diretto di kspaceFirstOrder3D (non il wrapper generico)
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
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
CFL_NUMBER: Final[float] = 0.3        # Courant–Friedrichs–Lewy (stabilità FDTD)
PML_SIZE: Final[int] = 10             # Spessore strato PML [voxels] (perfectly matched layer)
PML_ALPHA: Final[float] = 2.0         # Attenuazione PML [Np/m] — default k-Wave
SOURCE_MAGNITUDE: Final[float] = 1.0  # Ampiezza della sorgente [Pa]
KWAVE_HDF5_PREFIX: Final[str] = "kwave_sim_"
 
 
# ---------------------------------------------------------------------------
# Eccezione custom GPU
# ---------------------------------------------------------------------------
 
class GPUInitializationError(RuntimeError):
    """
    Sollevata quando il backend CUDA / binario kspaceFirstOrder3DC non è
    disponibile o fallisce l'inizializzazione.
 
    Il sistema NON esegue fallback silenzioso su CPU: su HPC ogni job
    deve girare sulla GPU assegnata dal job scheduler (SLURM).
    Abortire esplicitamente evita ore di calcolo CPU non pianificate.
 
    Attributes
    ----------
    reason : str
        Descrizione tecnica del motivo del fallimento.
    """
 
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
    """
    Parametri di configurazione per la simulazione FDTD k-Wave.
 
    Tutti i valori hanno default sensati per uso HPC, ma possono essere
    sovrascritti per tuning specifico di singole stanze.
 
    Parameters
    ----------
    source_pos_m : Tuple[float, float, float]
        Posizione della sorgente (x, y, z) in metri.
    mic_pos_m : Tuple[float, float, float]
        Posizione del microfono (x, y, z) in metri.
    fs_simulation : float
        Frequenza di campionamento della simulazione [Hz].
        Tipicamente derivata dal passo temporale CFL: fs = 1/dt.
        Settata automaticamente da KWaveEngine se lasciata a 0.
    t_end_s : float
        Durata della simulazione [s]. Default = 1.0 s.
    pml_size : int
        Spessore del PML in voxel. Default = 10.
    use_gpu : bool
        Se True (default), forza il backend CUDA.
        Se False, usa CPU (solo per test locali — NON per HPC).
    hdf5_dir : Path | None
        Directory per i file HDF5 temporanei di k-Wave.
        Se None, usa tempfile.gettempdir().
    smooth_source : bool
        Se True, applica lo smoothing della sorgente suggerito da k-Wave
        per ridurre le discontinuità iniziali nel campo.
    """
 
    source_pos_m:   Tuple[float, float, float]
    mic_pos_m:      Tuple[float, float, float]
    fs_simulation:  float = 0.0          # calcolato automaticamente
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
    """
    Motore FDTD basato su k-wave-python per la generazione della RIR
    a bassa frequenza (RIR_LF).
 
    Uso tipico
    ----------
    >>> physics = PhysicsTranslator.translate_profile(room_profile)
    >>> sim_params = KWaveSimParams(
    ...     source_pos_m=(1.0, 1.0, 1.0),
    ...     mic_pos_m=(3.0, 2.0, 1.5),
    ... )
    >>> engine = KWaveEngine(physics_params=physics, sim_params=sim_params)
    >>> RIR_LF = engine.run()
 
    Notes
    -----
    Il metodo run() gestisce internamente:
      - Costruzione griglia
      - Pittura del medium (rho/c per ogni voxel di boundary)
      - Snap-to-grid di sorgente e microfono
      - Generazione impulso band-limited
      - Esecuzione FDTD GPU
      - Estrazione e normalizzazione RIR_LF
      - Pulizia file HDF5 temporanei
    """
 
    def __init__(
        self,
        physics_params: dict,
        sim_params: KWaveSimParams,
    ) -> None:
        """
        Parameters
        ----------
        physics_params : dict
            Output di PhysicsTranslator.translate_profile() (Modulo 1).
            Chiavi richieste: f_schroeder, dx, grid_shape, rho_per_surface,
            volume, t60.
        sim_params : KWaveSimParams
            Parametri di configurazione della simulazione.
        """
        self._validate_physics_params(physics_params)
        self.p  = physics_params
        self.sp = sim_params
 
        # Estratti dal dict Modulo 1 per accesso rapido
        self.dx:  float               = physics_params["dx"]
        self.f_s: float               = physics_params["f_schroeder"]
        self.Nx, self.Ny, self.Nz     = physics_params["grid_shape"]
        self.rho_walls: dict[str, float] = physics_params["rho_per_surface"]
 
        # Calcolato dal CFL
        self.dt: float  = CFL_NUMBER * self.dx / C_AIR
        self.Nt: int    = int(np.ceil(sim_params.t_end_s / self.dt))
        self.fs_sim: float = 1.0 / self.dt
 
        logger.info(
            "KWaveEngine inizializzato: grid=(%d x %d x %d) | "
            "dx=%.4f m | dt=%.3e s | Nt=%d | fs_sim=%.1f Hz",
            self.Nx, self.Ny, self.Nz,
            self.dx, self.dt, self.Nt, self.fs_sim,
        )
 
    # ------------------------------------------------------------------ #
    #  Validazione                                                         #
    # ------------------------------------------------------------------ #
 
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
 
    # ------------------------------------------------------------------ #
    #  GPU check                                                           #
    # ------------------------------------------------------------------ #
 
    def _enforce_gpu(self) -> SimulationExecutionOptions:
        """
        Verifica la disponibilità GPU e costruisce le SimulationExecutionOptions
        per il backend CUDA di k-Wave.
 
        Strategy di detection (in ordine):
          1. Controlla che il modulo 'pycuda' o 'torch.cuda' sia importabile
             (proxy per la presenza del driver NVIDIA).
          2. Controlla che il binario kspaceFirstOrder3DC sia nel PATH o
             nella variabile d'ambiente KWAVE_BIN_PATH.
          3. Costruisce SimulationExecutionOptions con is_gpu_simulation=True.
 
        Returns
        -------
        SimulationExecutionOptions
            Opzioni di esecuzione con GPU abilitata.
 
        Raises
        ------
        GPUInitializationError
            Se qualsiasi controllo fallisce.
        """
        if not self.sp.use_gpu:
            # Modalità CPU esplicita (solo per sviluppo/debug locale)
            logger.warning(
                "use_gpu=False: esecuzione su CPU. "
                "NON usare questa modalità su HPC."
            )
            return SimulationExecutionOptions(is_gpu_simulation=False)
 
        # --- Step 1: verifica driver NVIDIA via pycuda o torch ---
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
            pass  # pycuda non installato — proviamo torch
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
                pass  # torch non installato
 
        if not cuda_available:
            raise GPUInitializationError(
                "Impossibile verificare la disponibilità CUDA: "
                "né pycuda né torch sono installati. "
                "Installa almeno uno dei due come proxy per il driver NVIDIA:\n"
                "    pip install pycuda   oppure   pip install torch"
            )
 
        # --- Step 2: verifica binario kspaceFirstOrder3DC ---
        kwave_bin = self._find_kwave_binary()
 
        # FIX: indentazione corretta (era sfasata di 1 spazio extra)
        logger.info(
            "GPU enforcement OK [%s] | binario k-Wave: %s",
            cuda_source, kwave_bin,
        )
 
        # --- Step 3: costruisce opzioni GPU ---
        env_path = os.environ.get("KWAVE_BIN_PATH", "")
        exec_options = SimulationExecutionOptions(
            is_gpu_simulation=True,
            binary_dir=env_path if env_path else None,
        )
        return exec_options
 
    @staticmethod
    def _find_kwave_binary() -> str:
        """
        Localizza il binario kspaceFirstOrder3DC richiesto da k-Wave per
        l'esecuzione GPU/CPU off-process.
 
        Cerca in ordine:
          1. Variabile d'ambiente KWAVE_BIN_PATH
          2. PATH di sistema (shutil.which)
          3. Directory corrente
 
        Returns
        -------
        str
            Percorso assoluto del binario.
 
        Raises
        ------
        GPUInitializationError
            Se il binario non viene trovato.
        """
        binary_name = "kspaceFirstOrder3DC"
 
        # 1. Variabile d'ambiente
        env_path = os.environ.get("KWAVE_BIN_PATH", "")
        if env_path:
            candidate = Path(env_path) / binary_name
            if candidate.is_file():
                return str(candidate)
            logger.warning(
                "KWAVE_BIN_PATH='%s' impostato ma binario non trovato in quella path.",
                env_path,
            )
 
        # 2. PATH di sistema
        which_result = shutil.which(binary_name)
        if which_result:
            return which_result
 
        # 3. Directory corrente
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
 
    # ------------------------------------------------------------------ #
    #  Costruzione griglia                                                 #
    # ------------------------------------------------------------------ #
 
    def _build_grid(self) -> kWaveGrid:
        """
        Costruisce la kWaveGrid con dimensioni (Nx, Ny, Nz) e passo dx.
 
        La griglia include implicitamente il PML (Perfectly Matched Layer)
        alle 6 facce: k-Wave lo gestisce internamente, ma il numero di voxel
        Nx/Ny/Nz che passiamo è quello della stanza fisica (senza PML),
        poiché kspaceFirstOrder3D aggiunge automaticamente il PML in base
        a SimulationOptions.pml_size.
 
        Returns
        -------
        kWaveGrid
        """
        grid = kWaveGrid(
            [self.Nx, self.Ny, self.Nz],
            [self.dx, self.dx, self.dx],
        )
        logger.info(
            "kWaveGrid costruita: [%d, %d, %d] voxel | dx=%.4f m",
            self.Nx, self.Ny, self.Nz, self.dx,
        )
        return grid
 
    # ------------------------------------------------------------------ #
    #  Pittura del medium                                                  #
    # ------------------------------------------------------------------ #
 
    def _build_medium(self) -> kWaveMedium:
        """
        Costruisce il kWaveMedium 3D assegnando densità (rho) e velocità
        del suono (c) a ogni voxel della griglia.
 
        Strategia di "painting"
        -----------------------
        - Tutti i voxel interni (aria) ricevono rho=RHO_AIR, c=C_AIR.
        - I voxel di boundary (1 voxel di spessore alle 6 facce) ricevono
          la rho_wall calcolata dal Modulo 1 per quella superficie,
          mantenendo c=C_AIR (l'impedenza differenziale è concentrata su rho).
 
        Layout degli assi nella griglia k-Wave:
          asse 0 (i) → X → lunghezza → wall_front (i=0), wall_back (i=Nx-1)
          asse 1 (j) → Y → larghezza → wall_left  (j=0), wall_right (j=Ny-1)
          asse 2 (k) → Z → altezza   → floor       (k=0), ceiling   (k=Nz-1)
 
        Returns
        -------
        kWaveMedium
            Medium con array 3D di rho e c omogeneo per c.
        """
        logger.info("Costruzione medium 3D (%d x %d x %d)...", self.Nx, self.Ny, self.Nz)
 
        # Inizializza tutto come aria
        rho_map = np.full((self.Nx, self.Ny, self.Nz), RHO_AIR,  dtype=np.float32)
        c_map   = np.full((self.Nx, self.Ny, self.Nz), C_AIR,    dtype=np.float32)
 
        # Pittura boundary — 1 voxel di spessore per ciascuna faccia
        # Asse 0 (X): wall_front @ i=0, wall_back @ i=Nx-1
        rho_map[0,  :, :] = self.rho_walls["wall_front"]
        rho_map[-1, :, :] = self.rho_walls["wall_back"]
 
        # Asse 1 (Y): wall_left @ j=0, wall_right @ j=Ny-1
        rho_map[:, 0,  :] = self.rho_walls["wall_left"]
        rho_map[:, -1, :] = self.rho_walls["wall_right"]
 
        # Asse 2 (Z): floor @ k=0, ceiling @ k=Nz-1
        rho_map[:, :, 0 ] = self.rho_walls["floor"]
        rho_map[:, :, -1] = self.rho_walls["ceiling"]
 
        # Statistiche di sanità
        n_boundary = (
            2 * self.Ny * self.Nz +   # wall_front + wall_back
            2 * self.Nx * self.Nz +   # wall_left + wall_right
            2 * self.Nx * self.Ny     # floor + ceiling
        )
        n_interior = self.Nx * self.Ny * self.Nz - n_boundary
        logger.info(
            "Medium painted: %d voxel interni (aria) | %d voxel boundary",
            n_interior, n_boundary,
        )
        logger.debug(
            "rho range: [%.2f, %.2f] kg/m3",
            float(rho_map.min()), float(rho_map.max()),
        )
 
        medium = kWaveMedium(
            sound_speed=c_map,
            density=rho_map,
        )
        return medium
 
    # ------------------------------------------------------------------ #
    #  Snap-to-grid                                                        #
    # ------------------------------------------------------------------ #
 
    def _snap_to_grid(
        self,
        pos_m: Tuple[float, float, float],
        label: str,
    ) -> Tuple[int, int, int]:
        """
        Converte coordinate metriche (x, y, z) in indici di griglia interi,
        assicurandosi che il punto cada all'interno dei voxel interni
        (non sui voxel di boundary, che hanno proprietà del muro).
 
        Formula
        -------
            idx = round(pos_m / dx)
            idx = clip(idx, 1, N-2)   ← esclude il boundary layer
 
        Parameters
        ----------
        pos_m : Tuple[float, float, float]
            Posizione in metri (x, y, z).
        label : str
            Etichetta per il logging ("source" o "mic").
 
        Returns
        -------
        Tuple[int, int, int]
            Indici di griglia (ix, iy, iz), 0-based.
 
        Raises
        ------
        ValueError
            Se la posizione è fuori dalla stanza (> dimensione fisica).
        """
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
 
        # Warning se il punto è stato spostato di più di mezzo voxel
        shift_m = np.sqrt(
            (ix * self.dx - x_m) ** 2 +
            (iy * self.dx - y_m) ** 2 +
            (iz * self.dx - z_m) ** 2
        )
        if shift_m > 0.5 * self.dx:
            logger.warning(
                "Snap-to-grid '%s': shift %.4f m (> dx/2 = %.4f m). "
                "Considera dimensioni di stanza/griglia più fini.",
                label, shift_m, 0.5 * self.dx,
            )
        else:
            logger.info(
                "Snap-to-grid '%s': (%.3f, %.3f, %.3f) m -> voxel (%d, %d, %d) "
                "| shift=%.4f m",
                label, x_m, y_m, z_m, ix, iy, iz, shift_m,
            )
 
        return ix, iy, iz
 
    # ------------------------------------------------------------------ #
    #  Segnale sorgente band-limited                                        #
    # ------------------------------------------------------------------ #
 
    def _build_source_signal(self) -> NDArray[np.float64]:
        """
        Genera un segnale sorgente impulsivo band-limited per la simulazione
        FDTD, limitato alla frequenza di Schroeder f_s per evitare aliasing
        spaziale sulla griglia.
 
        Strategia a due stadi
        ---------------------
        Stage 1 — Gaussiana modulata (chirp tone burst):
            s(t) = A * exp(-((t - t0) / sigma)^2) * sin(2π f_c t)
 
            con f_c = f_s / 2 (frequenza centrale), sigma tale che la
            larghezza a -6dB copra [DC, f_s].
            Questo garantisce un rise time fisicamente plausibile e un
            contenuto in frequenza concentrato nella banda di interesse.
 
        Stage 2 — Filtro Butterworth passa-basso (ordine 8, zero-phase):
            Rimozione di qualsiasi energia residua oltre f_s, causata
            dalla natura finita della finestra Gaussiana.
            Usa sosfiltfilt per zero-phase (no distorsione di fase).
 
        Il segnale finale è normalizzato a picco unitario.
 
        Returns
        -------
        NDArray[np.float64]
            Array 1D, lunghezza Nt, campionato a fs_sim = 1/dt.
        """
        t = np.arange(self.Nt) * self.dt          # vettore temporale [s]
        f_c = self.f_s / 2.0                       # frequenza centrale [Hz]
 
        # --- Stage 1: tone burst Gaussiano ---
        # sigma: la finestra copre ~4 periodi a f_c
        n_cycles = 4
        sigma = n_cycles / (2.0 * np.pi * f_c)
        t0    = 3.0 * sigma                        # offset per evitare taglio in t=0
 
        envelope = np.exp(-((t - t0) / sigma) ** 2)
        carrier  = np.sin(2.0 * np.pi * f_c * t)
        signal   = SOURCE_MAGNITUDE * envelope * carrier
 
        logger.info(
            "Source signal — Gaussiana modulata: f_c=%.2f Hz, sigma=%.5f s, "
            "t0=%.5f s, Nt=%d",
            f_c, sigma, t0, self.Nt,
        )
 
        # --- Stage 2: low-pass Butterworth a f_s (zero-phase) ---
        nyq = self.fs_sim / 2.0
        if self.f_s < nyq:
            sos = butter(
                N=8,
                Wn=self.f_s / nyq,
                btype="low",
                output="sos",
            )
            signal = sosfiltfilt(sos, signal)
            logger.info(
                "Source signal — LP Butterworth ord.8 a f_s=%.2f Hz applicato.",
                self.f_s,
            )
        else:
            logger.warning(
                "f_s (%.2f Hz) >= Nyquist (%.2f Hz): filtro LP non applicato.",
                self.f_s, nyq,
            )
 
        # Normalizzazione picco unitario
        peak = np.max(np.abs(signal))
        if peak > 1e-12:
            signal /= peak
 
        logger.debug(
            "Source signal: range=[%.4f, %.4f], energy=%.4f",
            float(signal.min()), float(signal.max()),
            float(np.sum(signal ** 2)),
        )
        return signal.astype(np.float64)
 
    # ------------------------------------------------------------------ #
    #  Setup sorgente e sensore                                            #
    # ------------------------------------------------------------------ #
 
    def _build_source(
        self,
        source_idx: Tuple[int, int, int],
        signal: NDArray[np.float64],
    ) -> kSource:
        """
        Costruisce il kSource con una sorgente puntuale alla posizione
        snap-to-grid e il segnale band-limited come time series di pressione.
 
        Parameters
        ----------
        source_idx : Tuple[int, int, int]
            Indici (ix, iy, iz) del voxel sorgente.
        signal : NDArray[np.float64]
            Segnale sorgente 1D, lunghezza Nt.
 
        Returns
        -------
        kSource
        """
        ix, iy, iz = source_idx
 
        # k-Wave usa maschere booleane 3D per la posizione sorgente
        source_mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        source_mask[ix, iy, iz] = True
 
        source = kSource()
        source.p_mask = source_mask
        # p deve essere shape (n_sources, Nt) = (1, Nt)
        source.p = signal[np.newaxis, :]
 
        logger.info(
            "kSource: voxel=(%d, %d, %d) | segnale shape=%s",
            ix, iy, iz, source.p.shape,
        )
        return source
 
    def _build_sensor(
        self,
        mic_idx: Tuple[int, int, int],
    ) -> kSensor:
        """
        Costruisce il kSensor puntuale alla posizione snap-to-grid.
        Registra solo la pressione (campo scalare) per minimizzare
        l'I/O e la RAM di output.
 
        Parameters
        ----------
        mic_idx : Tuple[int, int, int]
            Indici (ix, iy, iz) del voxel microfono.
 
        Returns
        -------
        kSensor
        """
        ix, iy, iz = mic_idx
 
        sensor_mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        sensor_mask[ix, iy, iz] = True
 
        sensor = kSensor()
        sensor.mask = sensor_mask
        sensor.record = ["p"]     # registra solo pressione
 
        logger.info(
            "kSensor: voxel=(%d, %d, %d)",
            ix, iy, iz,
        )
        return sensor
 
    # ------------------------------------------------------------------ #
    #  Gestione file HDF5 temporanei                                        #
    # ------------------------------------------------------------------ #
 
    def _make_hdf5_paths(self) -> Tuple[Path, Path]:
        """
        Genera percorsi unici per i file HDF5 di input/output di k-Wave.
 
        Usa un UUID per garantire l'unicità anche su filesystem condivisi
        (NFS su cluster HPC con più job paralleli sulla stessa directory).
 
        Returns
        -------
        Tuple[Path, Path]
            (input_hdf5_path, output_hdf5_path)
        """
        hdf5_dir = self.sp.hdf5_dir or Path(tempfile.gettempdir())
        hdf5_dir.mkdir(parents=True, exist_ok=True)
 
        uid = uuid.uuid4().hex[:12]
        input_path  = hdf5_dir / f"{KWAVE_HDF5_PREFIX}{uid}_input.h5"
        output_path = hdf5_dir / f"{KWAVE_HDF5_PREFIX}{uid}_output.h5"
 
        logger.info(
            "HDF5 temporanei: input=%s | output=%s",
            input_path, output_path,
        )
        return input_path, output_path
 
    @staticmethod
    def _cleanup_hdf5(*paths: Path) -> None:
        """
        Rimuove i file HDF5 temporanei in modo sicuro.
        Chiamato in finally per garantire la pulizia anche in caso di crash.
        """
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
                    logger.debug("HDF5 eliminato: %s", p)
            except OSError as e:
                logger.warning("Impossibile eliminare HDF5 %s: %s", p, e)
 
    # ------------------------------------------------------------------ #
    #  Post-processing RIR                                                 #
    # ------------------------------------------------------------------ #
 
    @staticmethod
    def _extract_rir(sensor_data: dict) -> NDArray[np.float64]:
        """
        Estrae e normalizza la RIR_LF dall'output del sensore k-Wave.
 
        k-Wave restituisce un dict con chiave 'p' contenente un array
        di shape (n_sensors, Nt). Per il sensore singolo prendiamo
        la prima (e unica) riga e normalizziamo al picco.
 
        Parameters
        ----------
        sensor_data : dict
            Output di kspaceFirstOrder3D. Atteso: {'p': ndarray (1, Nt)}.
 
        Returns
        -------
        NDArray[np.float64]
            RIR_LF normalizzata, shape (Nt,).
        """
        p_raw = sensor_data.get("p", None)
        if p_raw is None:
            raise RuntimeError(
                "sensor_data non contiene la chiave 'p'. "
                "Verifica che sensor.record = ['p'] sia impostato correttamente."
            )
 
        # (1, Nt) -> (Nt,)
        rir = np.array(p_raw, dtype=np.float64).squeeze()
 
        if rir.ndim != 1:
            raise RuntimeError(
                f"RIR_LF attesa 1D, ottenuto shape={rir.shape}. "
                f"Sensor multi-punto non supportato in questo modulo."
            )
 
        # Normalizzazione al picco
        peak = np.max(np.abs(rir))
        if peak < 1e-15:
            logger.warning(
                "RIR_LF ha picco quasi nullo (%.2e). "
                "La simulazione potrebbe non aver convergito correttamente.",
                peak,
            )
        else:
            rir /= peak
 
        logger.info(
            "RIR_LF estratta: Nt=%d campioni | peak originale=%.4e | "
            "range normalizzato=[%.4f, %.4f]",
            len(rir), peak, float(rir.min()), float(rir.max()),
        )
        return rir
 
    # ------------------------------------------------------------------ #
    #  Entry point pubblico                                                #
    # ------------------------------------------------------------------ #
 
    def run(self) -> NDArray[np.float64]:
        """
        Esegue l'intera pipeline FDTD k-Wave e restituisce la RIR_LF.
 
        Sequenza
        --------
        1.  _enforce_gpu()          — verifica CUDA / lancia GPUInitializationError
        2.  _build_grid()           — costruisce kWaveGrid
        3.  _build_medium()         — pittura rho/c su griglia 3D
        4.  snap-to-grid sorgente   — coordinate metriche → indici voxel
        5.  snap-to-grid microfono
        6.  _build_source_signal()  — impulso Gaussiano + LP Butterworth
        7.  _build_source()         — kSource con maschera 3D
        8.  _build_sensor()         — kSensor puntuale
        9.  _make_hdf5_paths()      — percorsi UUID per file temporanei
        10. kspaceFirstOrder3D()    — simulazione FDTD GPU
        11. _extract_rir()          — normalizzazione e conversione 1D
        12. _cleanup_hdf5()         — eliminazione file HDF5 (in finally)
 
        Returns
        -------
        NDArray[np.float64]
            RIR_LF: array 1D float64, lunghezza Nt, normalizzato a picco 1.
            Campionamento: fs_sim = 1/dt = C_AIR / (CFL * dx) [Hz].
 
        Raises
        ------
        GPUInitializationError
            CUDA non disponibile o binario kspaceFirstOrder3DC non trovato.
        MemoryError
            Se k-Wave non riesce ad allocare la griglia (catturato e re-raise).
        RuntimeError
            Qualsiasi altro fallimento interno di k-Wave.
        """
        logger.info("=" * 60)
        logger.info("KWaveEngine.run() — START")
        logger.info(
            "  Source: %s m | Mic: %s m | t_end=%.3f s | GPU=%s",
            self.sp.source_pos_m, self.sp.mic_pos_m,
            self.sp.t_end_s, self.sp.use_gpu,
        )
        logger.info("=" * 60)
 
        # --- Step 1: GPU enforcement ---
        exec_options = self._enforce_gpu()
 
        # --- Step 2-3: Griglia e medium ---
        grid   = self._build_grid()
        medium = self._build_medium()
 
        # --- Step 4-5: Snap-to-grid ---
        src_idx = self._snap_to_grid(self.sp.source_pos_m, "source")
        mic_idx = self._snap_to_grid(self.sp.mic_pos_m,    "mic")
 
        if src_idx == mic_idx:
            logger.warning(
                "Source e microfono cadono nello stesso voxel (%s). "
                "La RIR sarà dominata dall'impulso diretto senza separazione "
                "spaziale. Considera posizioni diverse.",
                src_idx,
            )
 
        # --- Step 6-8: Segnale e trasduttori ---
        signal = self._build_source_signal()
        source = self._build_source(src_idx, signal)
        sensor = self._build_sensor(mic_idx)
 
        # --- Step 9: Path HDF5 ---
        input_hdf5, output_hdf5 = self._make_hdf5_paths()
 
        # --- Step 10-11: Simulazione FDTD (con cleanup garantito) ---
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
                data_cast="single",         # float32 per ridurre RAM GPU
            )
 
            # FIX 1: kspaceFirstOrder3D invece di kspaceFirstOrder
            # FIX 2: ordine posizionale obbligatorio (kgrid, source, sensor, medium)
            sensor_data = kspaceFirstOrder3D(
                kgrid,
                source,
                sensor,
                medium,
                simulation_options=sim_options,
                execution_options=exec_options,
            )
 
            logger.info("Simulazione k-Wave completata.")
 
        except MemoryError:
            logger.error(
                "MemoryError durante la simulazione k-Wave. "
                "Griglia (%d x %d x %d) troppo grande per la GPU disponibile. "
                "Aumenta dx o riduci la stanza.",
                self.Nx, self.Ny, self.Nz,
            )
            raise
 
        except Exception as e:
            logger.error(
                "Simulazione k-Wave fallita: %s: %s",
                type(e).__name__, e,
            )
            raise RuntimeError(
                f"kspaceFirstOrder3D ha sollevato {type(e).__name__}: {e}"
            ) from e
 
        finally:
            # Pulizia garantita dei file HDF5 temporanei
            self._cleanup_hdf5(input_hdf5, output_hdf5)
            logger.info("File HDF5 temporanei rimossi.")
 
        # --- Step 12: Estrazione RIR ---
        rir_lf = self._extract_rir(sensor_data)
 
        logger.info(
            "KWaveEngine.run() DONE — RIR_LF shape=%s | fs_sim=%.2f Hz",
            rir_lf.shape, self.fs_sim,
        )
        return rir_lf
 
 
# ---------------------------------------------------------------------------
# Funzione di convenienza — entry point da pipeline esterna
# ---------------------------------------------------------------------------
 
def generate_rir_lf(
    physics_params: dict,
    source_pos_m:   Tuple[float, float, float],
    mic_pos_m:      Tuple[float, float, float],
    t_end_s:        float = 1.0,
    use_gpu:        bool  = True,
    hdf5_dir:       Optional[Path] = None,
) -> Tuple[NDArray[np.float64], float]:
    """
    Funzione di convenienza che avvolge KWaveEngine per uso diretto dalla
    pipeline principale (Modulo 4 — Crossover).
 
    Parameters
    ----------
    physics_params : dict
        Output di PhysicsTranslator.translate_profile() [Modulo 1].
    source_pos_m : Tuple[float, float, float]
        Posizione sorgente in metri (x, y, z).
    mic_pos_m : Tuple[float, float, float]
        Posizione microfono in metri (x, y, z).
    t_end_s : float
        Durata della simulazione [s]. Default = 1.0.
    use_gpu : bool
        Se True (default HPC), enforcement GPU.
    hdf5_dir : Path | None
        Directory per file HDF5 temporanei.
 
    Returns
    -------
    Tuple[NDArray[np.float64], float]
        (RIR_LF, fs_simulation)
        - RIR_LF       : array 1D normalizzato
        - fs_simulation: frequenza di campionamento della simulazione [Hz]
                         (necessaria al Modulo 4 per il resampling)
    """
    sim_params = KWaveSimParams(
        source_pos_m=source_pos_m,
        mic_pos_m=mic_pos_m,
        t_end_s=t_end_s,
        use_gpu=use_gpu,
        hdf5_dir=hdf5_dir,
    )
 
    engine = KWaveEngine(
        physics_params=physics_params,
        sim_params=sim_params,
    )
 
    rir_lf = engine.run()
    return rir_lf, engine.fs_sim
 
 
# ---------------------------------------------------------------------------
# Self-test — verifica struttura senza eseguire la simulazione FDTD
# (k-wave-python potrebbe non essere installato nella macchina di sviluppo)
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
 
    # --- Costruisci profilo con Modulo 1 ---
    from m1_physics_setup import RoomAcousticProfile, PhysicsTranslator, SURFACE_KEYS
 
    room = RoomAcousticProfile(
        length=6.0,
        width=4.0,
        height=3.0,
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
        profile=room,
        target_freq_hz=125,
        max_ram_gb=16.0,
    )
 
    print(f"\nFisica Modulo 1:")
    print(f"  f_schroeder = {physics['f_schroeder']:.2f} Hz")
    print(f"  dx          = {physics['dx']*100:.3f} cm")
    print(f"  grid_shape  = {physics['grid_shape']}")
 
    # --- Costruisci engine senza eseguire run() ---
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
 
    # --- Test snap-to-grid ---
    print("\nTest snap-to-grid:")
    src_idx = engine._snap_to_grid((1.0, 1.0, 1.0), "source")
    mic_idx = engine._snap_to_grid((4.5, 3.0, 1.5), "mic")
    print(f"  Source: (1.0, 1.0, 1.0) m -> voxel {src_idx}")
    print(f"  Mic:    (4.5, 3.0, 1.5) m -> voxel {mic_idx}")
 
    # --- Test segnale sorgente ---
    print("\nTest source signal:")
    signal = engine._build_source_signal()
    print(f"  Shape:   {signal.shape}")
    print(f"  Peak:    {float(np.max(np.abs(signal))):.6f}")
    print(f"  Energy:  {float(np.sum(signal**2)):.6f}")
    # Verifica che non ci sia energia significativa oltre f_s
    fft_mag  = np.abs(np.fft.rfft(signal))
    freqs    = np.fft.rfftfreq(len(signal), d=engine.dt)
    f_s      = physics['f_schroeder']
    e_below  = float(np.sum(fft_mag[freqs <= f_s] ** 2))
    e_above  = float(np.sum(fft_mag[freqs >  f_s] ** 2))
    ratio_db = 10 * np.log10(e_above / (e_below + 1e-15))
    print(f"  Energia sopra f_s vs sotto: {ratio_db:.1f} dB (atteso << 0 dB)")
 
    # --- Test medium painting ---
    print("\nTest medium painting:")
    medium = engine._build_medium()
    print(f"  rho shape: {medium.density.shape}")
    print(f"  rho floor (k=0):    {medium.density[1, 1, 0]:.4f} kg/m3")
    print(f"  rho ceiling (k=-1): {medium.density[1, 1, -1]:.4f} kg/m3")
    print(f"  rho interior:       {medium.density[3, 3, 3]:.4f} kg/m3")
 
    # --- Test GPUInitializationError ---
    print("\nTest GPUInitializationError (atteso: eccezione su sistema senza GPU):")
    try:
        engine._enforce_gpu()
        print("  GPU disponibile — nessuna eccezione sollevata.")
    except GPUInitializationError as e:
        print(f"  [OK] GPUInitializationError: {str(e).splitlines()[0]}")
 
    print("\n" + "=" * 60)
    print("Self test Modulo 2 completato.")
    print("=" * 60)
 




