"""
build_graph.py — Convert a precinct shapefile into a GerryChain dual graph.

Run once per shapefile; the resulting JSON is cached in data/graphs/.

Usage:
    uv run python scripts/build_graph.py --chamber house
    uv run python scripts/build_graph.py --chamber senate
    uv run python scripts/build_graph.py --chamber congress
    uv run python scripts/build_graph.py --file data/raw/ga_vtd.shp --district-col HDIST
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd

from gerrychain import Graph


# ---------------------------------------------------------------------------
# Expected column names in the precinct shapefile.
# Edit these if your file uses different column names.
# ---------------------------------------------------------------------------
REQUIRED_COLS = {
    "population": ["TOTPOP_H", "TOTPOP_S", "TOTPOP_C", "TOTAL_POP", "TOTPOP", "TOT_POP"],
    "dem_votes":  ["DEM_VOTES", "G22DEM", "P20DEM", "VEST_DEM"],
    "rep_votes":  ["REP_VOTES", "G22REP", "P20REP", "VEST_REP"],
    "district":   ["HDIST", "SDIST", "CDIST", "DISTRICT", "DIST_ID"],
}

CHAMBER_DEFAULTS = {
    "house":    {"district_col": "HDIST",  "pop_col": "TOTPOP_H", "graph_out": "data/graphs/ga_house.json"},
    "senate":   {"district_col": "SDIST",  "pop_col": "TOTPOP_S", "graph_out": "data/graphs/ga_senate.json"},
    "congress": {"district_col": "CDIST",  "pop_col": "TOTPOP_C", "graph_out": "data/graphs/ga_congress.json"},
}

# prep_data.py produces this single ready-to-use file
READY_FILE = Path("data/raw/ga_precincts_ready.shp")

DATA_RAW = Path("data/raw")


def find_col(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name found in the GeoDataFrame."""
    cols = set(gdf.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def build_graph(shapefile: Path, district_col: str, out_path: Path) -> None:
    print(f"Loading shapefile: {shapefile}")
    gdf = gpd.read_file(shapefile)
    print(f"  {len(gdf)} precincts loaded")

    # Validate / report column presence
    for role, candidates in REQUIRED_COLS.items():
        found = find_col(gdf, candidates)
        if found:
            print(f"  [{role}] → '{found}'")
        else:
            print(f"  WARNING: no column found for '{role}'. "
                  f"Expected one of: {candidates}")

    if district_col not in gdf.columns:
        sys.exit(f"ERROR: district column '{district_col}' not found. "
                 f"Available: {list(gdf.columns)}")

    # Build adjacency graph (rook = shared boundary, no diagonal touch)
    print("Building dual graph (rook adjacency)…")
    graph = Graph.from_geodataframe(gdf, adjacency="rook")
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Check for disconnected components (islands)
    import networkx as nx
    components = list(nx.connected_components(graph))
    if len(components) > 1:
        print(f"  WARNING: {len(components)} disconnected components found. "
              "You may need to merge island precincts manually.")
    else:
        print("  Graph is fully connected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_json(str(out_path))
    print(f"Graph saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GerryChain dual graph from shapefile")
    parser.add_argument("--chamber", choices=["house", "senate", "congress"],
                        help="Preset chamber (uses defaults for GA)")
    parser.add_argument("--file", type=Path,
                        help="Path to shapefile (overrides chamber preset)")
    parser.add_argument("--district-col", default=None,
                        help="Column name for district assignment")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for graph JSON")
    args = parser.parse_args()

    if args.chamber:
        defaults = CHAMBER_DEFAULTS[args.chamber]
        shapefile = args.file or READY_FILE
        district_col = args.district_col or defaults["district_col"]
        out_path = args.out or Path(defaults["graph_out"])
    elif args.file:
        if not args.district_col:
            sys.exit("ERROR: --district-col is required when using --file")
        shapefile = args.file
        district_col = args.district_col
        out_path = args.out or Path("data/graphs") / (args.file.stem + ".json")
    else:
        parser.print_help()
        sys.exit(1)

    if not shapefile.exists():
        sys.exit(f"ERROR: shapefile not found: {shapefile}\n"
                 f"Download GA VTD data and place it in {DATA_RAW}/")

    build_graph(shapefile, district_col, out_path)


if __name__ == "__main__":
    main()
