import argparse
import glob
import os
import sqlite3

import h5py


def collect_h5_paths(base_dir):
    h5_dir = os.path.join(base_dir, "h5")
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"h5 folder not found in {base_dir}")
    return sorted(glob.glob(os.path.join(h5_dir, "*.h5")))


def iter_instances(h5_path):
    with h5py.File(h5_path, "r") as h5_file:
        for instance_key in h5_file.keys():
            yield instance_key


def initialize_db(db_path, h5_paths):
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        inserted = 0
        skipped = 0

        for h5_path in h5_paths:
            abs_h5_path = os.path.abspath(h5_path)
            for instance_key in iter_instances(h5_path):
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO region_matching_status
                    (h5_path, instance_key, status, region_matching, error)
                    VALUES (?, ?, 'pending', NULL, NULL)
                    """,
                    (abs_h5_path, instance_key),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1

        conn.commit()
        return inserted, skipped
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Populate region_matching_status with h5_path and instance_key."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing the h5 folder",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="dataset/region_matching.db",
        help="Path to the SQLite database file",
    )
    args = parser.parse_args()

    h5_paths = collect_h5_paths(args.base_dir)
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {os.path.join(args.base_dir, 'h5')}")

    inserted, skipped = initialize_db(args.db_path, h5_paths)
    print(f"Initialized database: {args.db_path}")
    print(f"H5 files scanned: {len(h5_paths)}")
    print(f"Rows inserted: {inserted}")
    print(f"Rows skipped: {skipped}")


if __name__ == "__main__":
    main()
