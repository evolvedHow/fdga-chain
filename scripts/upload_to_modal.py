"""
upload_to_modal.py — Upload data files to the Modal persistent volume.

Uploads only what the API needs to serve requests (~65 MB total).
Does NOT upload the 3 GB GerryChain input shapefiles (ga_pl2020_*).

Run from the fdga-chain repo root:
    python scripts/upload_to_modal.py

Re-run any time you generate new ensemble results locally:
    uv run python scripts/run_ensemble.py --chamber house --steps 10000
    python scripts/upload_to_modal.py        # sync the new parquet to Modal
    modal deploy modal_app.py                # redeploy if code also changed
"""

import subprocess
import sys
from pathlib import Path

VOLUME = "fdga-chain-data"

# Files to upload: (local_path, volume_path)
# volume_path is relative to the volume root.
# Volume is mounted at /data, so volume_path "ensembles" → /data/ensembles.
UPLOAD_DIRS = [
    ("data/ensembles",  "ensembles"),   # parquet + meta + stability JSON
    ("data/states",     "states"),      # config.json per state
    ("data/graphs",     "graphs"),      # pre-built GerryChain graphs (optional)
]

# Individual raw files — only the ones the API actually reads for map endpoints.
# Excludes the massive ga_pl2020_*.shp/dbf files (GerryChain input, not needed to serve).
RAW_FILES = [
    "ga_precincts_ready.shp",
    "ga_precincts_ready.dbf",
    "ga_precincts_ready.shx",
    "ga_precincts_ready.prj",
    "ga_precincts_ready.cpg",
    "House-2023 shape.shp",
    "House-2023 shape.dbf",
    "House-2023 shape.shx",
    "House-2023 shape.prj",
    "Senate-2023 shape file.shp",
    "Senate-2023 shape file.dbf",
    "Senate-2023 shape file.shx",
    "Senate-2023 shape file.prj",
    "Congress-2023 shape.shp",
    "Congress-2023 shape.dbf",
    "Congress-2023 shape.shx",
    "Congress-2023 shape.prj",
]


def run(cmd: list[str]) -> bool:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ✗ failed (exit {result.returncode})")
        return False
    return True


def check_modal():
    result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: modal not found. Install with: pip install modal")
        sys.exit(1)


def main():
    check_modal()
    repo_root = Path(__file__).parent.parent
    errors = []

    print(f"\nUploading data to Modal volume: {VOLUME}")
    print("=" * 60)

    # Upload whole directories
    for local_rel, vol_path in UPLOAD_DIRS:
        local = repo_root / local_rel
        if not local.exists():
            print(f"\n  SKIP (not found): {local_rel}")
            continue
        size_mb = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"\n  Uploading {local_rel}/  ({size_mb:.1f} MB) → /{vol_path}")
        ok = run(["modal", "volume", "put", "--force", VOLUME, str(local), f"/{vol_path}"])
        if not ok:
            errors.append(local_rel)

    # Upload selective raw files
    raw_dir = repo_root / "data" / "raw"
    existing_raw = [f for f in RAW_FILES if (raw_dir / f).exists()]
    missing_raw  = [f for f in RAW_FILES if not (raw_dir / f).exists()]

    if existing_raw:
        total_mb = sum((raw_dir / f).stat().st_size for f in existing_raw) / 1024 / 1024
        print(f"\n  Uploading {len(existing_raw)} raw shapefile components  ({total_mb:.1f} MB) → /raw/")
        for fname in existing_raw:
            src = raw_dir / fname
            ok = run(["modal", "volume", "put", "--force", VOLUME, str(src), f"/raw/{fname}"])
            if not ok:
                errors.append(f"raw/{fname}")
    else:
        print("\n  SKIP: no raw shapefiles found in data/raw/")

    if missing_raw:
        print(f"\n  Note: {len(missing_raw)} raw files not found locally (may not be needed):")
        for f in missing_raw:
            print(f"    - {f}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"✗ {len(errors)} upload(s) failed: {errors}")
        sys.exit(1)
    else:
        print("✓ Upload complete.")
        print("\nNext steps:")
        print("  1. modal run modal_app.py::list_volume   ← verify contents")
        print("  2. modal deploy modal_app.py              ← deploy the API")
        print("  3. Copy the printed URL into frontend/index.html → window.API_BASE_OVERRIDE")


if __name__ == "__main__":
    main()
