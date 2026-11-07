"""Inspect safetensors head shape without torch or loading arbitrary pickle files."""
import argparse
import json
import struct
from pathlib import Path

def inspect(folder):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    shapes = {}
    for path in folder.glob("*.safetensors"):
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 100_000_000:
                raise ValueError("Unreasonable safetensors header size")
            header = json.loads(handle.read(size))
        shapes.update({name: entry["shape"] for name,entry in header.items() if "classifier" in name and isinstance(entry,dict)})
    names = config.get("id2label", {})
    print(json.dumps({"class_names":names,"head_shapes":shapes,
                     "generic_labels":any(value.startswith("LABEL_") for value in names.values()),
                     "note":"Shapes and names do not prove training label provenance."},indent=2))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("folder")
    inspect(parser.parse_args().folder)
