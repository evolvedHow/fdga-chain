"""
prep_data.py — Prepare a single ready-to-use precinct shapefile for GerryChain.

Inputs (all in data/raw/):
  ga_gen_20_st_prec.shp      — precinct boundaries + 2020 statewide election results (RDH)
  ga_vest_20.shp             — fallback precinct file (VEST)
  House-2023 shape.shp       — enacted House district polygons (180 districts)
  Senate-2023 shape file.shp — enacted Senate district polygons (56 districts)
  Congress-2023 shape.shp    — enacted Congressional district polygons (14 districts)

  Population (one of these, auto-detected):
    ga_pl2020_p1_b.shp       — Census block PL 94-171 P1 table (TOTPOP + race)
    ga_pl2020_p4_b.shp       — Census block PL 94-171 P4 table (Hispanic VAP)
    ga_vtd_pl94.shp          — VTD-level PL 94-171 (if available)

Output:
  data/raw/ga_precincts_ready.shp — single file with all columns GerryChain needs

Usage:
    uv run python scripts/prep_data.py
    uv run python scripts/prep_data.py --pop-file data/raw/ga_vtd_pl94.shp --pop-col P0010001
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW = Path("data/raw")

# Column name candidates for total population in VTD-level files
POP_COL_CANDIDATES = [
    "P0010001",
    "TOTPOP20", "TOTPOP", "TOT_POP", "POPULATION", "POP20",
]

# Demographic columns for VTD-level files: output name → candidate source columns
DEMO_COLS = {
    "BVAP": ["P0050004", "P0030004", "BVAP20", "NH18_BLK"],
    "HVAP": ["P0040003", "HVAP20", "H18_POP"],
}


def find_col(gdf, candidates):
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def spatial_join_district(precincts, districts, out_col, district_col="DISTRICT"):
    """
    Assign each precinct to a district via centroid-in-polygon join.
    Falls back to largest-overlap for precincts that span district boundaries.
    """
    districts = districts.to_crs(precincts.crs)
    centroids = precincts.copy()
    centroids["geometry"] = precincts.geometry.centroid

    joined = gpd.sjoin(
        centroids[["geometry"]],
        districts[[district_col, "geometry"]],
        how="left",
        predicate="within",
    )

    unmatched = joined[joined[district_col].isna()].index
    if len(unmatched) > 0:
        print(f"  {len(unmatched)} precincts unmatched by centroid — using largest overlap…")
        for idx in unmatched:
            precinct_geom = precincts.loc[idx, "geometry"]
            overlaps = districts.copy()
            overlaps["_overlap"] = overlaps.geometry.intersection(precinct_geom).area
            best = overlaps.loc[overlaps["_overlap"].idxmax(), district_col]
            joined.loc[idx, district_col] = best

    result = precincts.copy()
    result[out_col] = joined[district_col].values
    print(f"  → {out_col}: {result[out_col].nunique()} unique districts assigned")
    return result


def aggregate_blocks_to_precincts(precincts):
    """
    Aggregate Census block-level PL 94-171 data to precinct level.

    Uses ga_pl2020_p1_b.shp (P1 table) for TOTPOP and Black alone population.
    Uses ga_pl2020_p4_b.shp (P4 table) for Hispanic VAP.

    BVAP = Black alone (total population), used as proxy for Black VAP.
    This slightly over-estimates minority population (includes under-18),
    which is acceptable for educational ensemble analysis.
    """
    p1_file = RAW / "ga_pl2020_p1_b.shp"
    p4_file = RAW / "ga_pl2020_p4_b.shp"

    if not p1_file.exists():
        return None

    print(f"  Loading {p1_file.name} ({p1_file.stat().st_size // 1_000_000} MB)…")
    blocks = gpd.read_file(str(p1_file))
    blocks = blocks[["GEOID20", "P0010001", "P0010004", "geometry"]].copy()
    blocks.columns = ["GEOID20", "TOTPOP", "BLACK_ALONE", "geometry"]
    blocks = blocks.to_crs(precincts.crs)
    print(f"  {len(blocks):,} Census blocks loaded  (GA total pop: {blocks['TOTPOP'].sum():,})")

    # Load Hispanic VAP from P4
    if p4_file.exists():
        print(f"  Loading {p4_file.name} for Hispanic VAP…")
        p4 = gpd.read_file(str(p4_file))[["GEOID20", "P0040003"]].copy()
        p4.columns = ["GEOID20", "HVAP"]
        blocks = blocks.merge(p4, on="GEOID20", how="left")
        blocks["HVAP"] = blocks["HVAP"].fillna(0)
    else:
        print("  WARNING: ga_pl2020_p4_b.shp not found — HVAP will be 0")
        blocks["HVAP"] = 0

    # Use Black alone total pop as BVAP proxy
    blocks["BVAP"] = blocks["BLACK_ALONE"]

    # Block centroids for spatial join
    print(f"  Joining {len(blocks):,} block centroids → {len(precincts):,} precincts…")
    block_cents = blocks.copy()
    block_cents["geometry"] = blocks.geometry.centroid

    prec_indexed = precincts[["geometry"]].copy().reset_index().rename(columns={"index": "prec_idx"})

    joined = gpd.sjoin(
        block_cents[["GEOID20", "TOTPOP", "BVAP", "HVAP", "geometry"]],
        prec_indexed,
        how="left",
        predicate="within",
    )

    unmatched = joined["prec_idx"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} blocks didn't match any precinct centroid — excluded")

    # Sum by precinct index
    agg = joined.groupby("prec_idx")[["TOTPOP", "BVAP", "HVAP"]].sum()

    result = precincts.copy()
    result["TOTPOP"] = result.index.map(agg["TOTPOP"]).fillna(0).astype(int)
    result["BVAP"]   = result.index.map(agg["BVAP"]).fillna(0).astype(int)
    result["HVAP"]   = result.index.map(agg["HVAP"]).fillna(0).astype(int)

    zero_pop = (result["TOTPOP"] == 0).sum()
    if zero_pop:
        print(f"  WARNING: {zero_pop} precincts have zero population after aggregation")

    return result


def join_vtd_population(precincts, pop_file, pop_col):
    """Join VTD-level population data to precincts via spatial join (1-to-1)."""
    print(f"  Loading from {pop_file} (column: {pop_col})…")
    pop_gdf = gpd.read_file(str(pop_file)).to_crs(precincts.crs)

    centroids = precincts[["geometry"]].copy()
    centroids["geometry"] = precincts.geometry.centroid

    extra_cols = [pop_col]
    for candidates in DEMO_COLS.values():
        c = find_col(pop_gdf, candidates)
        if c and c not in extra_cols:
            extra_cols.append(c)

    joined = gpd.sjoin(
        centroids,
        pop_gdf[extra_cols + ["geometry"]],
        how="left",
        predicate="within",
    )

    result = precincts.copy()
    result["TOTPOP"] = joined[pop_col].fillna(0).values

    for alias, candidates in DEMO_COLS.items():
        c = find_col(pop_gdf, candidates)
        if c:
            result[alias] = joined[c].fillna(0).values
        else:
            result[alias] = 0

    return result


def make_rescaled_pop_col(precincts, district_col, district_file, out_col, dist_id_col="DISTRICT"):
    """
    Create a new population column by rescaling block-aggregated TOTPOP within each district
    so the district total matches the official count in the enacted district shapefile.

    Each chamber (House, Senate, Congress) gets its own column (TOTPOP_H, TOTPOP_S, TOTPOP_C)
    because the three sets of districts have different boundaries.
    """
    districts = gpd.read_file(str(district_file))
    if "POPULATION" not in districts.columns:
        print(f"  WARNING: No POPULATION column in {district_file.name} — using raw TOTPOP")
        precincts[out_col] = precincts["TOTPOP"]
        return precincts

    official_pops = dict(zip(districts[dist_id_col], districts["POPULATION"]))

    result = precincts.copy()
    result[out_col] = result["TOTPOP"].astype(float)

    adjusted = 0
    for dist_id, official_pop in official_pops.items():
        mask = result[district_col] == dist_id
        computed_pop = result.loc[mask, "TOTPOP"].sum()
        if computed_pop > 0 and official_pop > 0:
            scale = official_pop / computed_pop
            if abs(scale - 1) > 0.001:
                result.loc[mask, out_col] = (result.loc[mask, "TOTPOP"] * scale).round()
                adjusted += 1

    result[out_col] = result[out_col].fillna(0).astype(int)
    print(f"  {out_col}: rescaled {adjusted} districts  (total: {result[out_col].sum():,})")
    return result


def find_vtd_population_file():
    """Auto-detect a VTD-level population file in data/raw/."""
    candidates = [
        "ga_vtd_pl94.shp",
        "ga_pl94_vtd.shp",
        "ga_2020_vtd_pl94171.shp",
        "ga_vtd20.shp",
    ]
    for name in candidates:
        p = RAW / name
        if p.exists():
            gdf = gpd.read_file(str(p))
            col = find_col(gdf, POP_COL_CANDIDATES)
            if col:
                print(f"  Auto-detected VTD population file: {p} (column: {col})")
                return p, col
    return None, None


def prep(pop_file=None, pop_col=None):

    # ------------------------------------------------------------------ #
    # 1. Load base precinct file                                           #
    # ------------------------------------------------------------------ #
    base_file = RAW / "ga_gen_20_st_prec.shp"
    fallback   = RAW / "ga_vest_20.shp"

    if base_file.exists():
        print(f"Loading precincts from {base_file.name}…")
        precincts = gpd.read_file(str(base_file))
    elif fallback.exists():
        print(f"Loading precincts from {fallback.name} (fallback)…")
        precincts = gpd.read_file(str(fallback))
    else:
        raise FileNotFoundError(
            "No precinct file found. Expected ga_gen_20_st_prec.shp in data/raw/"
        )

    print(f"  {len(precincts)} precincts loaded")

    # Normalize CRS
    if precincts.crs is None:
        print("  CRS missing — assuming EPSG:4019, converting to EPSG:4326")
        precincts = precincts.set_crs("EPSG:4019").to_crs("EPSG:4326")
    else:
        precincts = precincts.to_crs("EPSG:4326")

    # ------------------------------------------------------------------ #
    # 2. Standardize election columns                                      #
    # ------------------------------------------------------------------ #
    print("\nStandardizing election columns…")
    if "G20PREDBID" in precincts.columns and "G20PRERTRU" in precincts.columns:
        precincts["DEM_VOTES"] = precincts["G20PREDBID"]
        precincts["REP_VOTES"] = precincts["G20PRERTRU"]
        print("  Primary: 2020 Presidential (Biden / Trump)")
    else:
        print("  WARNING: 2020 Presidential columns not found — setting DEM/REP_VOTES to 0")
        precincts["DEM_VOTES"] = 0
        precincts["REP_VOTES"] = 0

    # ------------------------------------------------------------------ #
    # 3. Population data                                                   #
    # ------------------------------------------------------------------ #
    print("\nLooking for population data…")

    pop_joined = False

    # Priority 1: explicit --pop-file argument
    if pop_file is not None and pop_col is not None:
        precincts = join_vtd_population(precincts, pop_file, pop_col)
        pop_joined = True

    # Priority 2: block-level PL 94-171 files (ga_pl2020_p1_b.shp)
    elif (RAW / "ga_pl2020_p1_b.shp").exists():
        print("  Found block-level PL 94-171 files — aggregating blocks → precincts…")
        result = aggregate_blocks_to_precincts(precincts)
        if result is not None:
            precincts = result
            pop_joined = True

    # Priority 3: VTD-level file (auto-detect)
    else:
        vtd_file, vtd_col = find_vtd_population_file()
        if vtd_file is not None:
            precincts = join_vtd_population(precincts, vtd_file, vtd_col)
            pop_joined = True

    if pop_joined:
        total_pop = precincts["TOTPOP"].sum()
        print(f"  Total population:   {total_pop:,.0f}")
        print(f"  House ideal pop:    {total_pop/180:,.0f}")
        print(f"  Senate ideal pop:   {total_pop/56:,.0f}")
        print(f"  Congress ideal pop: {total_pop/14:,.0f}")
    else:
        print()
        print("=" * 60)
        print("POPULATION DATA MISSING — cannot run GerryChain without it")
        print("=" * 60)
        print()
        print("You have block-level PL 94-171 files but they weren't found.")
        print("Expected: data/raw/ga_pl2020_p1_b.shp")
        print()
        print("Or download VTD-level data from RDH Georgia:")
        print("  'Georgia 2020 Census Redistricting Data (PL 94-171) – VTDs'")
        print("  Rename to: data/raw/ga_vtd_pl94.shp")
        print("=" * 60)
        print()
        print("Continuing with TOTPOP=0 placeholder for geometry validation only.")
        precincts["TOTPOP"] = 0
        precincts["BVAP"]   = 0
        precincts["HVAP"]   = 0

    # ------------------------------------------------------------------ #
    # 4. Spatial join: assign precincts → 2023 enacted districts           #
    # ------------------------------------------------------------------ #
    print("\nJoining 2023 enacted district assignments…")

    house    = gpd.read_file(str(RAW / "House-2023 shape.shp")).to_crs("EPSG:4326")
    senate   = gpd.read_file(str(RAW / "Senate-2023 shape file.shp")).to_crs("EPSG:4326")
    congress = gpd.read_file(str(RAW / "Congress-2023 shape.shp")).to_crs("EPSG:4326")

    print("  House (180)…")
    precincts = spatial_join_district(precincts, house,    out_col="HDIST")
    print("  Senate (56)…")
    precincts = spatial_join_district(precincts, senate,   out_col="SDIST")
    print("  Congressional (14)…")
    precincts = spatial_join_district(precincts, congress, out_col="CDIST")

    # ------------------------------------------------------------------ #
    # 4b. Per-chamber rescaled population columns                          #
    # Each chamber's TOTPOP_X is calibrated to that chamber's official     #
    # district populations. This ensures GerryChain's population           #
    # constraint is satisfied for each chamber independently.              #
    # ------------------------------------------------------------------ #
    if precincts["TOTPOP"].sum() > 0:
        print("\nCreating per-chamber population columns…")
        precincts = make_rescaled_pop_col(
            precincts, "HDIST", RAW / "House-2023 shape.shp", "TOTPOP_H"
        )
        precincts = make_rescaled_pop_col(
            precincts, "SDIST", RAW / "Senate-2023 shape file.shp", "TOTPOP_S"
        )
        precincts = make_rescaled_pop_col(
            precincts, "CDIST", RAW / "Congress-2023 shape.shp", "TOTPOP_C"
        )

    # ------------------------------------------------------------------ #
    # 5. Validate                                                          #
    # ------------------------------------------------------------------ #
    print("\nValidation summary:")
    print(f"  Precincts:          {len(precincts)}")
    print(f"  House districts:    {precincts['HDIST'].nunique()} / 180  "
          f"(missing: {precincts['HDIST'].isna().sum()})")
    print(f"  Senate districts:   {precincts['SDIST'].nunique()} / 56   "
          f"(missing: {precincts['SDIST'].isna().sum()})")
    print(f"  Congress districts: {precincts['CDIST'].nunique()} / 14   "
          f"(missing: {precincts['CDIST'].isna().sum()})")
    print(f"  Total population:   {precincts['TOTPOP'].sum():,.0f}")
    print(f"  DEM_VOTES total:    {precincts['DEM_VOTES'].sum():,.0f}")
    print(f"  REP_VOTES total:    {precincts['REP_VOTES'].sum():,.0f}")
    final_cols = [c for c in precincts.columns if c != "geometry"]
    print(f"  Output columns:     {final_cols}")

    # ------------------------------------------------------------------ #
    # 6. Save                                                              #
    # ------------------------------------------------------------------ #
    out_path = RAW / "ga_precincts_ready.shp"
    precincts.to_file(str(out_path))
    print(f"\nSaved: {out_path}")

    if precincts["TOTPOP"].sum() > 0:
        print("\nNext step:")
        print("  uv run python scripts/build_graph.py --chamber house")
        print("  uv run python scripts/build_graph.py --chamber senate")
        print("  uv run python scripts/build_graph.py --chamber congress")
    else:
        print("\nNext step: add population file (see instructions above), then re-run.")


def main():
    parser = argparse.ArgumentParser(description="Prepare precinct data for GerryChain")
    parser.add_argument("--pop-file", type=Path, default=None,
                        help="Path to population shapefile (auto-detected if omitted)")
    parser.add_argument("--pop-col", default=None,
                        help="Column name for total population (auto-detected if omitted)")
    args = parser.parse_args()
    prep(pop_file=args.pop_file, pop_col=args.pop_col)


if __name__ == "__main__":
    main()
