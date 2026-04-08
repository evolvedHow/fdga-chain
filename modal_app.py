"""
modal_app.py — Deploy fdga-chain API to Modal.

─── What goes where ───────────────────────────────────────────────────────────

  Container image (baked in at build time, from this repo):
    api/          ← FastAPI app + data_loader
    pyproject.toml

  Persistent volume  (uploaded once, updated when data changes):
    data/ensembles/    ← GerryChain parquet + meta + stability JSON   (~600 KB)
    data/states/       ← config.json per state                         (~12 KB)
    data/raw/          ← enacted district shapefiles + precinct file   (~50 MB)
    data/graphs/       ← pre-built GerryChain graphs (only for re-runs)(~13 MB)

  NOT uploaded (only needed for running GerryChain locally):
    data/raw/ga_pl2020_*.shp/dbf   ← 3 GB Census input files, not needed to serve

─── One-time setup ────────────────────────────────────────────────────────────

  1. Install Modal:
       pip install modal

  2. Authenticate:
       modal setup

  3. Create the secret (Mapbox token + any future env vars):
       modal secret create fdga-chain-secrets MAPBOX_TOKEN=pk.eyJ1...

  4. Upload data to the volume (run from the fdga-chain repo root):
       python scripts/upload_to_modal.py

  5. Deploy:
       modal deploy modal_app.py

  6. Copy the printed URL into frontend/index.html:
       window.API_BASE_OVERRIDE = 'https://evolvedhow--fdga-chain-api.modal.run';
       Then push to GitHub.

─── Day-to-day ───────────────────────────────────────────────────────────────

  Dev/preview (live-reload tunnel):
    modal serve modal_app.py

  Re-deploy after code changes:
    modal deploy modal_app.py

  Re-upload data after new ensemble runs:
    python scripts/upload_to_modal.py

  Check volume contents:
    modal run modal_app.py::list_volume

"""

import modal

# ---------------------------------------------------------------------------
# Image — system deps + Python deps + app source code
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    # GDAL needed by geopandas / fiona
    .apt_install("libgdal-dev", "gdal-bin", "libspatialindex-dev")
    # Python dependencies from pyproject.toml
    .pip_install_from_pyproject("pyproject.toml")
    # Bake the API source into the image
    # (re-runs on `modal deploy` whenever these files change)
    .copy_local_dir("api", "/root/api")
    .copy_local_file("pyproject.toml", "/root/pyproject.toml")
)

# ---------------------------------------------------------------------------
# Persistent volume — data files that outlive container restarts
# Mounted at /data inside the container.
# Paths inside volume match the local data/ layout, e.g.:
#   local:  data/ensembles/house_ensemble.parquet
#   volume: /data/ensembles/house_ensemble.parquet
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
    # Long timeout supports async GerryChain ensemble runs (~10 min max)
    timeout=600,
    # Keep one warm container — avoids cold-start delay on the education UI
    keep_warm=1,
    # Reads MAPBOX_TOKEN (and any future vars) from the named secret.
    # Create with: modal secret create fdga-chain-secrets MAPBOX_TOKEN=pk.xxx
    secrets=[modal.Secret.from_name("fdga-chain-secrets", required=False)],
)
@modal.asgi_app()
def api():
    import os
    import sys

    # Working dir = / so that Path("data/...") resolves to /data/... (volume mount)
    os.chdir("/")
    # Python path = /root so that `from api.main import ...` finds the baked-in code
    sys.path.insert(0, "/root")

    from api.main import app as fastapi_app
    return fastapi_app


# ---------------------------------------------------------------------------
# Utility: list volume contents
# Usage: modal run modal_app.py::list_volume
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=60,
)
def list_volume():
    """Print everything in the data volume with sizes."""
    import os
    from pathlib import Path

    data_dir = Path("/data")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("Volume /data is empty. Run: python scripts/upload_to_modal.py")
        return

    total_bytes = 0
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for f in sorted(files):
            fpath = Path(root) / f
            size = fpath.stat().st_size
            total_bytes += size
            rel = fpath.relative_to(data_dir)
            print(f"  {rel:<60}  {size/1024:>8.1f} kB")

    print(f"\n  Total: {total_bytes/1024/1024:.1f} MB")
