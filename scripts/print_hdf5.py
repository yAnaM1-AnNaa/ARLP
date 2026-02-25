import argparse
import json
import os
from typing import Any

import h5py
import numpy as np


def _format_value(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except Exception:
            return repr(value)
    if isinstance(value, np.ndarray):
        np.set_printoptions(threshold=np.inf, linewidth=200)
        return np.array2string(value, separator=', ')
    if np.isscalar(value):
        return repr(value)
    return repr(value)


def _write_attrs(f, attrs, indent: int):
    for key, val in attrs.items():
        f.write(' ' * indent + f'@attr {key}: {_format_value(val)}\n')


def _visit(name, obj, f):
    if isinstance(obj, h5py.Group):
        f.write(f'[GROUP] /{name}\n')
        _write_attrs(f, obj.attrs, indent=2)
    elif isinstance(obj, h5py.Dataset):
        f.write(f'[DATASET] /{name}\n')
        f.write(f'  shape: {obj.shape}\n')
        f.write(f'  dtype: {obj.dtype}\n')
        _write_attrs(f, obj.attrs, indent=2)
        try:
            data = obj[()]
            f.write('  data:\n')
            if isinstance(data, np.ndarray):
                data_str = _format_value(data)
                for line in data_str.splitlines():
                    f.write('    ' + line + '\n')
            else:
                f.write('    ' + _format_value(data) + '\n')
        except Exception as exc:
            f.write(f'  data: <failed to read: {exc}>\n')
    f.write('\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Dump all contents of an HDF5 file to a text output')
    parser.add_argument('--input', '-i', required=True, help='Path to input HDF5 file')
    parser.add_argument('--output', '-o', required=True, help='Path to output text file')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Input file not found: {args.input}')
        return 2

    with h5py.File(args.input, 'r') as h5f, open(args.output, 'w', encoding='utf-8') as out:
        out.write(f'HDF5: {args.input}\n\n')
        _write_attrs(out, h5f.attrs, indent=0)
        out.write('\n')
        h5f.visititems(lambda name, obj: _visit(name, obj, out))

    print(f'Wrote dump to: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
