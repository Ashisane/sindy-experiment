"""
witvliet_reader.py
------------------
c302-compatible connectome data reader for the Witvliet et al. 2021 (Nature)
C. elegans developmental connectome dataset (8 larval/adult stages).

Implements the interface expected by c302:
    read_data(include_nonconnected_cells=False) -> (sorted list[str], list[ConnectionInfo])
    read_muscle_data()                           -> ([], [], [])

Supports 8 developmental stages (stage=1 through stage=8).

Chemical synapses â€” number field:
    c302 uses 'number' as a raw synapse contact count.  The JSON stores
    individual contacts keyed by vast_id.  We aggregate (pre, post) pairs
    and count the number of distinct contacts (vast_ids), identical to how
    classic White-lab spreadsheet data reports synapse counts.

    A weighted_size attribute (nm^3) is also accumulated per pair for
    downstream strength analysis, but is NOT passed into ConnectionInfo
    (which would break c302 compatibility).

Gap junctions â€” deduplicate + one direction only:
    The JSON file stores each physical contact twice (n1â†’n2 and n2â†’n1).
    We keep the first occurrence per catmaid_id.  c302 treats GapJunctions
    as bidirectional internally.

Data directory: ../nature2021/data/synapses/ (relative to this file)
"""

import collections
import json
import os

from c302.ConnectomeReader import ConnectionInfo, PREFERRED_NEURON_NAMES

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Absolute path so this module works whether imported from mdg_build/ or
# from inside the c302 package directory (where it gets copied at runtime).
_THIS_FILE = os.path.abspath(__file__)
_CANDIDATES = [
    # When running from mdg_build/ directly
    os.path.normpath(os.path.join(os.path.dirname(_THIS_FILE), "..", "nature2021", "data", "synapses")),
    # When copied into c302 package (e.g. .venv/Lib/site-packages/c302/)
    os.path.normpath(os.path.join(os.path.dirname(_THIS_FILE), "..", "..", "..", "..", "Desktop", "mdg", "nature2021", "data", "synapses")),
    # Explicit absolute fallback
    r"C:\Users\UTKARSH\Desktop\mdg\nature2021\data\synapses",
]
DATA_DIR = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[-1])

# GABAergic neurons in C. elegans (White 1986 + later revisions)
_GABA_PREFIXES = ("VD", "DD")
_GABA_EXACT    = frozenset({"RID", "DVB", "RIS", "AVL"})

_PREFERRED = frozenset(PREFERRED_NEURON_NAMES)


def _synclass(pre: str) -> str:
    """Return the neurotransmitter class string for a presynaptic neuron."""
    if pre in _GABA_EXACT:
        return "GABA"
    for pfx in _GABA_PREFIXES:
        if pre.startswith(pfx):
            return "GABA"
    return "Acetylcholine"


# â”€â”€ Main reader class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class WitvlietDataReader:
    """
    Reads chemical synapses and gap junctions for one developmental stage
    from the Witvliet 2021 Nature dataset and converts them into c302
    ConnectionInfo objects.

    Parameters
    ----------
    stage : int
        Dataset index, 1â€“8 inclusive (default 1).
    """

    def __init__(self, stage: int = 1):
        if stage not in range(1, 9):
            raise ValueError(f"stage must be between 1 and 8, got {stage}")
        self.stage = stage
        self._syn_file = os.path.join(DATA_DIR, f"Dataset{stage}_synapses.json")
        self._gj_file  = os.path.join(DATA_DIR, f"Dataset{stage}_gapjunctions.json")

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def read_data(self, include_nonconnected_cells: bool = False):
        """
        Parse synapses and gap junctions and return c302-compatible output.

        Returns
        -------
        cells : sorted list[str]
            All neuron names participating in at least one accepted connection.
            If include_nonconnected_cells=True, all 302 canonical neurons are
            included.
        conns : list[ConnectionInfo]
            Chemical synapses (syntype='Send') and gap junctions
            (syntype='GapJunction').  For chemical synapses, number = count
            of distinct vast_id contacts sharing the same (pre, post) pair.
            For gap junctions, number = 1 per unique catmaid_id contact.
        """
        cells       = set()
        n_filtered  = 0

        # â”€â”€ Chemical synapses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Aggregate contacts per (pre, post) pair.
        # pair_count[pair]  = number of distinct vast_id contacts
        # pair_size[pair]   = total weighted nm^3 (informational only)
        pair_count  = collections.Counter()
        pair_size   = collections.defaultdict(float)
        pair_class  = {}   # synclass is constant per pre neuron

        with open(self._syn_file, "r", encoding="utf-8") as fh:
            synapses = json.load(fh)

        for entry in synapses:
            pre     = entry["pre"]
            posts   = entry["post"]
            weights = entry.get("post_weights") or [1] * len(posts)
            if len(weights) != len(posts):
                weights = [1] * len(posts)
            size    = entry.get("size", 0.0)

            for post_cell, weight in zip(posts, weights):
                if pre not in _PREFERRED or post_cell not in _PREFERRED:
                    n_filtered += 1
                    continue
                pair = (pre, post_cell)
                pair_count[pair] += 1
                pair_size[pair]  += size * weight
                if pair not in pair_class:
                    pair_class[pair] = _synclass(pre)
                cells.add(pre)
                cells.add(post_cell)

        chem_conns = []
        for pair, count in pair_count.items():
            pre, post_cell = pair
            chem_conns.append(
                ConnectionInfo(pre, post_cell, count, "Send", pair_class[pair])
            )

        # â”€â”€ Gap junctions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Keep first occurrence per catmaid_id (file stores both directions).
        gj_conns       = []
        seen_catmaid   = set()

        with open(self._gj_file, "r", encoding="utf-8") as fh:
            gapjunctions = json.load(fh)

        for entry in gapjunctions:
            cid = entry["catmaid_id"]
            n1  = entry["n1"]
            n2  = entry["n2"]

            if cid in seen_catmaid:
                continue
            seen_catmaid.add(cid)

            if n1 not in _PREFERRED or n2 not in _PREFERRED:
                n_filtered += 1
                continue

            gj_conns.append(
                ConnectionInfo(n1, n2, 1, "GapJunction", "Generic_GJ")
            )
            cells.add(n1)
            cells.add(n2)

        if include_nonconnected_cells:
            cells.update(PREFERRED_NEURON_NAMES)

        sorted_cells = sorted(cells)
        conns        = chem_conns + gj_conns
        n_chem       = len(chem_conns)
        n_gj         = len(gj_conns)

        print(
            f"[WitvlietReader] Dataset{self.stage}: "
            f"{len(sorted_cells)} neurons, "
            f"{n_chem} chemical synapses, "
            f"{n_gj} gap junctions | "
            f"filtered out: {n_filtered} entries"
        )

        return sorted_cells, conns

    def read_muscle_data(self):
        """
        The Witvliet 2021 dataset contains no muscle connectivity data.
        Returns three empty lists to satisfy the c302 reader interface.
        """
        return [], [], []


# â”€â”€ Module-level convenience functions (required by c302 reader convention) â”€â”€â”€

_DEFAULT_STAGE = 1
_instance: WitvlietDataReader | None = None


def set_stage(stage: int) -> WitvlietDataReader:
    """Set the module default stage and refresh the cached reader."""
    global _DEFAULT_STAGE, _instance
    if stage not in range(1, 9):
        raise ValueError(f"stage must be between 1 and 8, got {stage}")
    _DEFAULT_STAGE = stage
    if _instance is None or _instance.stage != stage:
        _instance = WitvlietDataReader(stage=stage)
    return _instance


def get_instance(stage: int | None = None) -> WitvlietDataReader:
    """Return (and cache) a module-level WitvlietDataReader instance."""
    global _instance
    stage = _DEFAULT_STAGE if stage is None else stage
    if _instance is None or _instance.stage != stage:
        _instance = WitvlietDataReader(stage=stage)
    return _instance


def read_data(include_nonconnected_cells: bool = False, stage: int | None = None):
    """Module-level wrapper for the currently selected stage."""
    return get_instance(stage=stage).read_data(
        include_nonconnected_cells=include_nonconnected_cells
    )


def read_muscle_data(stage: int | None = None):
    """Module-level wrapper — always returns empty muscle data."""
    return get_instance(stage=stage).read_muscle_data()

if __name__ == "__main__":
    print("=" * 65)
    print("WitvlietDataReader â€” standalone test")
    print("=" * 65)

    results = {}
    for stage in (1, 8):
        print(f"\n--- Stage {stage} ---")
        reader = WitvlietDataReader(stage=stage)
        cells, conns = reader.read_data()

        chem = [c for c in conns if c.syntype == "Send"]
        gj   = [c for c in conns if c.syntype == "GapJunction"]

        print(f"  Total ConnectionInfo objects : {len(conns)}")
        print(f"    Chemical (Send)            : {len(chem)}")
        print(f"    Gap junctions              : {len(gj)}")
        print(f"  Number range (chem)          : "
              f"min={min(c.number for c in chem)}, "
              f"max={max(c.number for c in chem)}")
        print("  First 5 ConnectionInfo objects:")
        for c in conns[:5]:
            print(
                f"    pre={c.pre_cell:<10}  post={c.post_cell:<10}  "
                f"num={c.number:>3}  type={c.syntype:<12}  class={c.synclass}"
            )
        results[stage] = (len(cells), len(chem), len(gj))

    print("\n" + "=" * 65)
    s1  = results[1]
    s8  = results[8]
    ok1 = 140 <= s1[0] <= 165
    ok8 = s8[0] > s1[0] and s8[1] > s1[1]
    print(f"Stage 1  neurons={s1[0]}  chem={s1[1]}  gj={s1[2]}  {'[OK]' if ok1 else '[WARN: outside 140-165]'}")
    print(f"Stage 8  neurons={s8[0]}  chem={s8[1]}  gj={s8[2]}  {'[OK - more than stage 1]' if ok8 else '[WARN: not more than stage 1]'}")
    print("=" * 65)


