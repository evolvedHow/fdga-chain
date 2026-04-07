"""
export_static.py — Pre-compute all API responses as static JSON files for GitHub Pages.

Run this after generating ensembles. The output goes to docs/ which GitHub Pages serves.
The frontend auto-detects static mode and reads from these files instead of the live API.

Usage:
    # Make sure the API is running on port 8001 first:
    uv run uvicorn api.main:app --port 8001 &

    # Then export:
    uv run python scripts/export_static.py

    # Commit docs/ to GitHub and enable Pages → Deploy from branch → /docs
    git add docs/
    git commit -m "Refresh static export"
    git push
"""

import json
import sys
import urllib.request
from pathlib import Path

API_BASE = "http://localhost:8001"
OUT_DIR  = Path("docs/data")
CHAMBERS = ["house", "senate", "congress"]
METRICS  = [
    "dem_seats", "efficiency_gap", "mean_median",
    "polsby_popper_mean", "polsby_popper_min",
    "majority_minority_districts", "num_cut_edges",
]


def fetch(path: str) -> dict | list | None:
    url = f"{API_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  WARN: {path} → {e}")
        return None


def save(data, filename: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(json.dumps(data, separators=(",", ":")))
    size = path.stat().st_size
    print(f"  ✓  {filename}  ({size/1024:.1f} kB)")


def main():
    print(f"Exporting static data to {OUT_DIR}/")
    print(f"API: {API_BASE}\n")

    # Check API is reachable
    health = fetch("/health")
    if health is None:
        print("ERROR: API not reachable. Start it first:")
        print("  uv run uvicorn api.main:app --port 8001")
        sys.exit(1)

    save(health, "health.json")
    available = health.get("chambers_ready", [])
    print(f"Chambers available: {available}\n")

    # Static endpoints
    for path, filename in [
        ("/api/chambers",       "chambers.json"),
        ("/api/info",           "info.json"),
        ("/api/algorithms",     "algorithms.json"),
        ("/api/ensemble/runs",  "runs.json"),
    ]:
        data = fetch(path)
        if data is not None:
            save(data, filename)

    # Per-chamber endpoints
    for chamber in available:
        print(f"\n── {chamber} ──")

        for path, filename in [
            (f"/api/ensemble/{chamber}/enacted",  f"{chamber}_enacted.json"),
            (f"/api/ensemble/{chamber}/summary",  f"{chamber}_summary.json"),
        ]:
            data = fetch(path)
            if data is not None:
                save(data, filename)

        # Histograms (one per metric)
        for metric in METRICS:
            data = fetch(f"/api/ensemble/{chamber}/histogram?metric={metric}&bins=40")
            if data is not None:
                save(data, f"{chamber}_hist_{metric}.json")

        # Map GeoJSON (can be large)
        data = fetch(f"/api/maps/{chamber}/enacted")
        if data is not None:
            save(data, f"{chamber}_map_enacted.json")

        data = fetch(f"/api/maps/{chamber}/stability")
        if data is not None and "detail" not in data:
            save(data, f"{chamber}_map_stability.json")
        else:
            print(f"  SKIP: {chamber}_map_stability.json (not available — re-run ensemble)")

    # Copy frontend HTML to docs/
    import shutil
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    src_html = Path("frontend/index.html")
    if src_html.exists():
        shutil.copy(src_html, docs_dir / "index.html")
        print(f"\n  ✓  docs/index.html  (copied from frontend/index.html)")
    # Ensure .nojekyll so GitHub Pages doesn't skip files starting with _
    (docs_dir / ".nojekyll").touch()

    print(f"\nDone. {len(list(OUT_DIR.glob('*.json')))} files written to {OUT_DIR}/")
    print("\nNext steps:")
    print("  1. git add docs/ && git commit -m 'Refresh static export' && git push")
    print("  2. Enable GitHub Pages: Settings → Pages → Deploy from branch → main → /docs")


if __name__ == "__main__":
    main()
