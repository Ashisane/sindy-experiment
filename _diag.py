import json
from collections import defaultdict
DATA_DIR = r"C:\Users\UTKARSH\Desktop\mdg\nature2021\data\synapses"
for stage in range(1, 9):
    fname = f"{DATA_DIR}\\Dataset{stage}_synapses.json"
    with open(fname) as f:
        data = json.load(f)
    pair_vast = defaultdict(set)
    ok = True
    for i, entry in enumerate(data):
        pre = entry.get("pre")
        if pre is None:
            print(f"D{stage} e{i}: no 'pre' field")
            ok = False
            continue
        uid = entry.get("catmaid_id")
        if uid is None:
            print(f"D{stage} e{i}: no catmaid_id, keys={list(entry.keys())}")
            uid = i
        try:
            for post in entry["post"]:
                pair_vast[(pre, post)].add(uid)
        except TypeError as e:
            print(f"D{stage} e{i}: TypeError {e} uid={type(uid).__name__}={uid}")
            ok = False
    print(f"D{stage}: {'OK' if ok else 'ERROR'} {len(pair_vast)} pairs")
