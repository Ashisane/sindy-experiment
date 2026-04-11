from __future__ import annotations

import importlib
import os
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
MDG_BUILD = REPO_ROOT / "mdg_build"
C302_ROOT = REPO_ROOT / "c302"
C302_PACKAGE = C302_ROOT / "c302"
OUT_SWEEP = MDG_BUILD / "output_sweep"
OUT_STAGES = MDG_BUILD / "output_stages"
OUT_SIM = MDG_BUILD / "output_sim"
OUT_C0 = MDG_BUILD / "output_c0"

T_BIO_HOURS = np.array([0, 5, 16, 27, 47, 70, 81, 120], dtype=float)
STIM_NEURONS = ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR"]
FIRED_THRESHOLD_MV = -20.0
SUBTHRESHOLD_THRESHOLD_MV = -40.0
DURATION_MS = 500.0
DT_MS = 0.05
STIM_DELAY_MS = 50.0
STIM_DURATION_MS = 400.0


@dataclass
class TraceSummary:
    n_timesteps: int
    n_neurons: int
    dt_ms: float
    min_voltage_mv: float
    max_voltage_mv: float
    fired: int
    subthreshold: int
    silent: int


@dataclass
class SimulationResult:
    stage: int
    amplitude_pa: float
    net_id: str
    target_directory: str
    dat_path: str
    lems_path: str
    neuron_order: list[str]
    summary: dict[str, Any]
    reused_existing: bool
    jneuroml_ok: bool


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ensure_import_paths() -> None:
    for path in (MDG_BUILD, C302_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def amp_to_tag(amplitude_pa: float) -> str:
    return str(amplitude_pa).replace(".", "p")


def expected_timesteps(duration_ms: float = DURATION_MS, dt_ms: float = DT_MS) -> int:
    return int(round(duration_ms / dt_ms)) + 1


def sync_witvliet_reader() -> Path:
    ensure_import_paths()
    source = MDG_BUILD / "witvliet_reader.py"
    dest = C302_PACKAGE / "witvliet_reader.py"
    if not dest.exists() or source.read_text(encoding="utf-8") != dest.read_text(encoding="utf-8"):
        shutil.copy2(source, dest)
    return dest


def import_local_witvliet_reader():
    ensure_import_paths()
    importlib.invalidate_caches()
    if "witvliet_reader" in sys.modules:
        return importlib.reload(sys.modules["witvliet_reader"])
    return importlib.import_module("witvliet_reader")


def import_c302_witvliet_reader(stage: int):
    ensure_import_paths()
    sync_witvliet_reader()
    importlib.invalidate_caches()
    if "c302.witvliet_reader" in sys.modules:
        mod = importlib.reload(sys.modules["c302.witvliet_reader"])
    else:
        mod = importlib.import_module("c302.witvliet_reader")
    mod.set_stage(stage)
    return mod


def load_stage_cells(stage: int) -> list[str]:
    reader_mod = import_local_witvliet_reader()
    cells, _ = reader_mod.WitvlietDataReader(stage=stage).read_data(
        include_nonconnected_cells=False
    )
    return cells


def configure_offset_current(params, amplitude_pa: float, source: str) -> None:
    params.set_bioparameter(
        "unphysiological_offset_current",
        f"{amplitude_pa} pA",
        source,
        "0",
    )
    params.set_bioparameter(
        "unphysiological_offset_current_del",
        f"{STIM_DELAY_MS} ms",
        source,
        "0",
    )
    params.set_bioparameter(
        "unphysiological_offset_current_dur",
        f"{STIM_DURATION_MS} ms",
        source,
        "0",
    )


@contextmanager
def pushd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def parse_output_columns(lems_path: Path, dat_name: str | None = None) -> list[str]:
    if not lems_path.exists():
        return []

    text = lems_path.read_text(encoding="utf-8")
    block = None
    if dat_name is not None:
        pattern = rf'<OutputFile[^>]*fileName="{re.escape(dat_name)}"[^>]*>(.*?)</OutputFile>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            block = match.group(1)
    if block is None:
        matches = re.findall(r"<OutputFile[^>]*>(.*?)</OutputFile>", text, re.DOTALL)
        for candidate in matches:
            if ".activity.dat" not in candidate:
                block = candidate
                break
    if block is None:
        return []

    columns = re.findall(r'<OutputColumn\s+id="([^"]+)"', block)
    return [col.replace("_v", "").replace("Pop_", "") for col in columns]


def load_voltage_data(dat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(dat_path)
    time_s = raw[:, 0]
    voltage_mv = raw[:, 1:] * 1000.0
    return time_s, voltage_mv


def summarize_voltage_data(dat_path: Path) -> TraceSummary:
    time_s, voltage_mv = load_voltage_data(dat_path)
    vmax = voltage_mv.max(axis=0)
    dt_ms = float((time_s[1] - time_s[0]) * 1000.0) if len(time_s) > 1 else float("nan")
    fired = int((vmax > FIRED_THRESHOLD_MV).sum())
    subthreshold = int(
        ((vmax >= SUBTHRESHOLD_THRESHOLD_MV) & (vmax <= FIRED_THRESHOLD_MV)).sum()
    )
    silent = int((vmax < SUBTHRESHOLD_THRESHOLD_MV).sum())
    return TraceSummary(
        n_timesteps=int(voltage_mv.shape[0]),
        n_neurons=int(voltage_mv.shape[1]),
        dt_ms=dt_ms,
        min_voltage_mv=float(voltage_mv.min()),
        max_voltage_mv=float(voltage_mv.max()),
        fired=fired,
        subthreshold=subthreshold,
        silent=silent,
    )


def validate_trace_file(
    dat_path: Path,
    expected_neurons: int | None = None,
    expected_steps: int | None = None,
) -> bool:
    if not dat_path.exists():
        return False
    try:
        summary = summarize_voltage_data(dat_path)
    except Exception:
        return False
    if expected_neurons is not None and summary.n_neurons != expected_neurons:
        return False
    if expected_steps is not None and summary.n_timesteps != expected_steps:
        return False
    return True


def run_jneuroml(lems_path: Path, target_directory: Path, verbose: bool = False) -> bool:
    ensure_import_paths()
    from pyneuroml import pynml

    with pushd(target_directory):
        return bool(
            pynml.run_lems_with_jneuroml(
                lems_path.name,
                nogui=True,
                load_saved_data=False,
                plot=False,
                max_memory="4G",
                verbose=verbose,
            )
        )


def run_stage_simulation(
    stage: int,
    amplitude_pa: float,
    net_id: str,
    target_directory: Path,
    *,
    duration_ms: float = DURATION_MS,
    dt_ms: float = DT_MS,
    cells_to_stimulate: list[str] | None = None,
    source_tag: str = "mdg_pipeline",
    reuse_existing: bool = False,
    verbose: bool = False,
) -> SimulationResult:
    ensure_import_paths()
    ensure_directory(target_directory)
    cells = load_stage_cells(stage)
    expected_neurons = len(cells)
    expected_steps = expected_timesteps(duration_ms=duration_ms, dt_ms=dt_ms)
    dat_path = target_directory / f"{net_id}.dat"
    lems_path = target_directory / f"LEMS_{net_id}.xml"

    if reuse_existing and validate_trace_file(dat_path, expected_neurons, expected_steps):
        neuron_order = parse_output_columns(lems_path, dat_path.name) or list(cells)
        summary = summarize_voltage_data(dat_path)
        return SimulationResult(
            stage=stage,
            amplitude_pa=amplitude_pa,
            net_id=net_id,
            target_directory=str(target_directory),
            dat_path=str(dat_path),
            lems_path=str(lems_path),
            neuron_order=neuron_order,
            summary=asdict(summary),
            reused_existing=True,
            jneuroml_ok=True,
        )

    import_c302_witvliet_reader(stage)
    import c302
    from c302 import parameters_C0

    params = parameters_C0.ParameterisedModel()
    configure_offset_current(params, amplitude_pa, source=source_tag)
    stim_cells = [cell for cell in (cells_to_stimulate or STIM_NEURONS) if cell in cells]

    for stale in (dat_path, target_directory / f"{net_id}.activity.dat"):
        if stale.exists():
            stale.unlink()

    c302.generate(
        net_id,
        params,
        data_reader="witvliet_reader",
        cells=cells,
        cells_to_stimulate=stim_cells,
        cells_to_plot=cells,
        duration=duration_ms,
        dt=dt_ms,
        target_directory=str(target_directory),
        verbose=verbose,
    )
    jneuroml_ok = run_jneuroml(lems_path, target_directory, verbose=verbose)
    if not dat_path.exists():
        raise FileNotFoundError(f"Expected trace file not found: {dat_path}")

    neuron_order = parse_output_columns(lems_path, dat_path.name) or list(cells)
    summary = summarize_voltage_data(dat_path)
    return SimulationResult(
        stage=stage,
        amplitude_pa=amplitude_pa,
        net_id=net_id,
        target_directory=str(target_directory),
        dat_path=str(dat_path),
        lems_path=str(lems_path),
        neuron_order=neuron_order,
        summary=asdict(summary),
        reused_existing=False,
        jneuroml_ok=jneuroml_ok,
    )


def locate_trace_file(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def neuron_to_class(name: str) -> str:
    trimmed = name[:-1] if len(name) > 2 and name[-1].upper() in "LR" else name
    trimmed = re.sub(r"\d+$", "", trimmed)
    return trimmed or "UNK"
