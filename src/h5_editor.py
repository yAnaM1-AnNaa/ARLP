"""
HDF5 dataset editor for ARLP training data.
root
    agveuv
        clip_similarities
        color_label_names
        color_name_features
        depth
Usage:
    # Show structure
    python src/h5_editor.py --h5_path file.h5
    python src/h5_editor.py --h5_path file.h5 --instance hdibix

    # Read a dataset
    python src/h5_editor.py --h5_path file.h5 --read hdibix/region_matching

    # Write from external file (.npy / .json / .txt)
    python src/h5_editor.py --h5_path file.h5 --write hdibix/embeddings_oai/Blue --data_path emb.npy
    python src/h5_editor.py --h5_path file.h5 --write hdibix/region_matching    --data_path rm.json
    python src/h5_editor.py --h5_path file.h5 --write hdibix/color_label_names  --data_path names.txt

    # Delete
    python src/h5_editor.py --h5_path file.h5 --delete hdibix/embeddings_st

    # Rename
    python src/h5_editor.py --h5_path file.h5 --rename hdibix/old_name hdibix/new_name

    # Copy
    python src/h5_editor.py --h5_path file.h5 --copy hdibix/embeddings_oai hdibix/embeddings_oai_bak

    # Batch text replace across all instances
    python src/h5_editor.py --h5_path file.h5 --replace region_matching "lean" "recline"
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np


# ------------------------------------------------------------------ #
#                          SHOW / INSPECT                             #
# ------------------------------------------------------------------ #

def _print_item(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"  {name}/")
    elif isinstance(obj, h5py.Dataset):
        extra = ""
        if obj.dtype == object or str(obj.dtype).startswith("|O"):
            try:
                val = obj[()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                s = str(val)
                if len(s) > 120:
                    s = s[:120] + "..."
                extra = f"  -> {s}"
            except Exception:
                pass
        print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}{extra}")


def do_show(h5_path, instance=None):
    with h5py.File(h5_path, "r") as f:
        if instance:
            if instance not in f:
                print(f"Error: instance '{instance}' not found", file=sys.stderr)
                return
            root = f[instance]
            print(f"=== {h5_path} / {instance} ===")
        else:
            root = f
            print(f"=== {h5_path} ===")
            # list top-level instances first
            for key in root:
                n = len(root[key].keys()) if isinstance(root[key], h5py.Group) else 0
                print(f"  {key}/  ({n} keys)")
            print()
        root.visititems(_print_item)


# ------------------------------------------------------------------ #
#                              READ                                   #
# ------------------------------------------------------------------ #

def do_read(h5_path, ds_path):
    with h5py.File(h5_path, "r") as f:
        if ds_path not in f:
            print(f"Error: '{ds_path}' not found", file=sys.stderr)
            return
        obj = f[ds_path]
        if isinstance(obj, h5py.Group):
            print(f"'{ds_path}' is a group, keys: {list(obj.keys())}")
            return
        val = obj[()]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if isinstance(val, str):
            try:
                print(json.dumps(json.loads(val), indent=2, ensure_ascii=False))
                return
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(val, np.ndarray):
            print(f"shape={val.shape}, dtype={val.dtype}")
            print(val if val.size <= 50 else f"{val.flat[:20]} ...")
        else:
            print(val)


# ------------------------------------------------------------------ #
#                             WRITE                                   #
# ------------------------------------------------------------------ #

def do_write(h5_path, ds_path, data_path):
    """
    Write external file into the H5 dataset at ds_path.
    File type is auto-detected by extension:
        .npy          -> numpy array
        .json         -> stored as UTF-8 bytes scalar (same as region_matching)
        .txt          -> one string per line -> variable-length string array
        anything else -> try numpy, fall back to raw bytes
    """
    if not os.path.isfile(data_path):
        print(f"Error: data file '{data_path}' not found", file=sys.stderr)
        return

    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".npy":
        data = np.load(data_path)
        kind = "array"
    elif ext == ".json":
        with open(data_path, "r", encoding="utf-8") as fp:
            text = fp.read()
        json.loads(text)  # validate
        data = text.encode("utf-8")
        kind = "json"
    elif ext == ".txt":
        with open(data_path, "r", encoding="utf-8") as fp:
            lines = [l.rstrip("\n") for l in fp if l.strip()]
        kind = "string_list"
        data = lines
    else:
        try:
            data = np.load(data_path)
            kind = "array"
        except Exception:
            with open(data_path, "rb") as fp:
                data = fp.read()
            kind = "bytes"

    with h5py.File(h5_path, "a") as f:
        # ensure parent group exists
        parts = ds_path.rsplit("/", 1)
        if len(parts) == 2:
            parent_path, ds_name = parts
            if parent_path not in f:
                f.create_group(parent_path)
            parent = f[parent_path]
        else:
            parent = f
            ds_name = parts[0]

        # remove old if exists
        if ds_name in parent:
            del parent[ds_name]

        if kind == "string_list":
            dt = h5py.special_dtype(vlen=str)
            ds = parent.create_dataset(ds_name, shape=(len(data),), dtype=dt)
            ds[:] = data
            print(f"Wrote {len(data)} strings -> '{ds_path}'")
        elif kind == "array":
            parent.create_dataset(ds_name, data=data)
            print(f"Wrote array {data.shape} {data.dtype} -> '{ds_path}'")
        else:
            parent.create_dataset(ds_name, data=data)
            print(f"Wrote {kind} ({len(data)} bytes) -> '{ds_path}'")


# ------------------------------------------------------------------ #
#                            DELETE                                   #
# ------------------------------------------------------------------ #

def do_delete(h5_path, ds_path):
    with h5py.File(h5_path, "a") as f:
        if ds_path not in f:
            print(f"Error: '{ds_path}' not found", file=sys.stderr)
            return
        del f[ds_path]
        print(f"Deleted '{ds_path}'")


# ------------------------------------------------------------------ #
#                            RENAME                                   #
# ------------------------------------------------------------------ #

def do_rename(h5_path, old_path, new_path):
    with h5py.File(h5_path, "a") as f:
        if old_path not in f:
            print(f"Error: '{old_path}' not found", file=sys.stderr)
            return
        f.move(old_path, new_path)
        print(f"Renamed '{old_path}' -> '{new_path}'")


# ------------------------------------------------------------------ #
#                             COPY                                    #
# ------------------------------------------------------------------ #

def do_copy(h5_path, src, dst):
    with h5py.File(h5_path, "a") as f:
        if src not in f:
            print(f"Error: '{src}' not found", file=sys.stderr)
            return
        if dst in f:
            print(f"Error: '{dst}' already exists, delete first", file=sys.stderr)
            return
        f.copy(src, dst)
        print(f"Copied '{src}' -> '{dst}'")


# ------------------------------------------------------------------ #
#                        REPLACE-TEXT                                 #
# ------------------------------------------------------------------ #

def do_replace(h5_path, field, old_text, new_text):
    count = 0
    with h5py.File(h5_path, "a") as f:
        for inst_key in f:
            grp = f[inst_key]
            if not isinstance(grp, h5py.Group) or field not in grp:
                continue
            val = grp[field][()]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            if not isinstance(val, str) or old_text not in val:
                continue
            new_val = val.replace(old_text, new_text)
            del grp[field]
            grp.create_dataset(field, data=new_val.encode("utf-8"))
            count += 1
            print(f"  [{inst_key}] replaced in {field}")
    print(f"Done. Modified {count} instance(s).")


# ------------------------------------------------------------------ #
#                           CLI PARSER                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="HDF5 editor for ARLP training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5_path", required=True, help="Path to the .h5 file")
    parser.add_argument("--instance", "-i", default=None,
                        help="Instance to inspect (used with show mode)")

    # operations (mutually exclusive)
    ops = parser.add_mutually_exclusive_group()
    ops.add_argument("--read", metavar="DS_PATH",
                     help="Read dataset at this H5 internal path")
    ops.add_argument("--write", metavar="DS_PATH",
                     help="Write to this H5 internal path (requires --data_path)")
    ops.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"),
                     help="Rename OLD path to NEW path")
    ops.add_argument("--copy", nargs=2, metavar=("SRC", "DST"),
                     help="Copy SRC to DST within the file")
    ops.add_argument("--replace", nargs=3, metavar=("FIELD", "OLD_TEXT", "NEW_TEXT"),
                     help="Batch replace text in FIELD across all instances")

    parser.add_argument("--data_path", default=None,
                        help="External file to write (.npy / .json / .txt)")
    args = parser.parse_args()

    # no operation flag -> default to show
    has_op = any([args.read, args.write, args.delete, args.rename,
                  args.copy, args.replace])

    if not has_op:
        do_show(args.h5_path, args.instance)
    elif args.read:
        do_read(args.h5_path, args.read)
    elif args.write:
        if not args.data_path:
            parser.error("--write requires --data_path")
        do_write(args.h5_path, args.write, args.data_path)
    elif args.delete:
        do_delete(args.h5_path, args.delete)
    elif args.rename:
        do_rename(args.h5_path, *args.rename)
    elif args.copy:
        do_copy(args.h5_path, *args.copy)
    elif args.replace:
        do_replace(args.h5_path, *args.replace)


if __name__ == "__main__":
    main()
