"""
modal_app.py — Deploy fdga-chain API to Modal (serverless containers).

Usage:
  # Development (tunnels localhost → public URL):
  modal serve modal_app.py

  # Production deploy:
  modal deploy modal_app.py

  # Upload data files to the persistent volume (run once, or after new ensemble runs):
  python scripts/upload_data_to_modal.py

After deploy, copy the printed URL (e.g. https://evolvedhow--fdga-chain-api.modal.run)
into frontend/index.html → the API_BASE_OVERRIDE line, then push to GitHub Pages.
"""

import modal

# ---------------------------------------------------------------------------
# Image — Python env with all dependencies
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    # GDAL system deps needed by geopandas/fiona
    .apt_install("libgdal-dev", "gdal-bin")
)

# ---------------------------------------------------------------------------
# Persistent volume — stores data/ directory across invocations
# Data files (shapefiles, parquet, GeoJSON) are uploaded separately via
# scripts/upload_data_to_modal.py
# ---------------------------------------------------------------------------

data_volume = modal.Volume.from_name("fdga-chain-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("fdga-chain")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    cpu=2,
    memory=4096,
    timeout=600,
    keep_warm=1,
    # Secrets: create once with `modal secret create fdga-chain-secrets MAPBOX_TOKEN=pk.xxx`
    # or via the Modal dashboard → Secrets.
    # Falls back gracefully if the secret doesn't exist yet.
    secrets=[modal.Secret.from_name("fdga-chain-secrets", required=False)],
)
@modal.asgi_app()
def api():
    import os
    import sys

    os.chdir("/")
    sys.path.insert(0, "/root")

    from api.main import app as fastapi_app
    return fastapi_app


# ---------------------------------------------------------------------------
# Helper: sync local data/ → Modal volume
# Run with: modal run modal_app.py::sync_data
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=300,
)
def sync_data():
    """
    Called from sync_data_to_modal.py to verify volume contents after upload.
    Prints a summary of what's in /data.
    """
    import os
    from pathlib import Path

    data_dir = Path("/data")
    if not data_dir.exists():
        print("Volume /data is empty.")
        return

    total_files = 0
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            fpath = Path(root) / f
            size_kb = fpath.stat().st_size / 1024
            rel = fpath.relative_to(data_dir)
            print(f"  {rel}  ({size_kb:.1f} kB)")
            total_files += 1
    print(f"\nTotal: {total_files} files in /data")
