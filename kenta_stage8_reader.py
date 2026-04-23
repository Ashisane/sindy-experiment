from __future__ import annotations

from typing import Iterable

try:
    from c302.witvliet_reader import WitvlietDataReader
except Exception:
    from witvliet_reader import WitvlietDataReader


_DEFAULT_STAGE = 8
_EXCLUDED_NEURONS: set[str] = set()
_instance = None


class LesionedWitvlietReader:
    def __init__(self, stage: int = 8, exclude_neurons: Iterable[str] | None = None):
        self.stage = stage
        self.exclude_neurons = set(exclude_neurons or [])
        self._base_reader = WitvlietDataReader(stage=stage)

    def read_data(self, include_nonconnected_cells: bool = False):
        cells, conns = self._base_reader.read_data(
            include_nonconnected_cells=include_nonconnected_cells
        )
        filtered_cells = sorted(cell for cell in cells if cell not in self.exclude_neurons)
        filtered_conns = [
            conn
            for conn in conns
            if conn.pre_cell not in self.exclude_neurons
            and conn.post_cell not in self.exclude_neurons
        ]
        return filtered_cells, filtered_conns

    def read_muscle_data(self):
        return [], [], []


def set_stage(stage: int) -> LesionedWitvlietReader:
    global _DEFAULT_STAGE, _instance
    _DEFAULT_STAGE = stage
    _instance = LesionedWitvlietReader(stage=stage, exclude_neurons=_EXCLUDED_NEURONS)
    return _instance


def set_excluded_neurons(exclude_neurons: Iterable[str] | None):
    global _EXCLUDED_NEURONS, _instance
    _EXCLUDED_NEURONS = set(exclude_neurons or [])
    _instance = LesionedWitvlietReader(stage=_DEFAULT_STAGE, exclude_neurons=_EXCLUDED_NEURONS)
    return _instance


def get_instance(stage: int | None = None, exclude_neurons: Iterable[str] | None = None):
    global _instance
    stage = _DEFAULT_STAGE if stage is None else stage
    if exclude_neurons is not None:
        return set_excluded_neurons(exclude_neurons)
    if _instance is None or _instance.stage != stage:
        _instance = LesionedWitvlietReader(stage=stage, exclude_neurons=_EXCLUDED_NEURONS)
    return _instance


def read_data(include_nonconnected_cells: bool = False, stage: int | None = None):
    return get_instance(stage=stage).read_data(
        include_nonconnected_cells=include_nonconnected_cells
    )


def read_muscle_data(stage: int | None = None):
    return get_instance(stage=stage).read_muscle_data()
