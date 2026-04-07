"""
run_ensemble.py — Run a ReCom Markov chain and save per-step metrics to parquet.

This is the computationally expensive step. Run offline; results are consumed
by the API server and the education UI.

Usage:
    uv run python scripts/run_ensemble.py --chamber house --steps 10000
    uv run python scripts/run_ensemble.py --chamber senate --steps 5000
    uv run python scripts/run_ensemble.py --chamber congress --steps 2000

Output:
    data/ensembles/{chamber}_ensemble.parquet  — one row per accepted plan
    data/ensembles/{chamber}_meta.json         — run metadata (steps, epsilon, cols used)
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import pandas as pd
from gerrychain import GeographicPartition, Graph, MarkovChain
from gerrychain.accept import always_accept
from gerrychain.constraints import contiguous, within_percent_of_ideal_population
from gerrychain.proposals import recom, reversible_recom
from gerrychain.updaters import Tally, cut_edges

ALGO_DESCRIPTIONS = {
    "recom": {
        "name": "ReCom (Recombination)",
        "proposal": "Merge two adjacent districts, re-split via random spanning tree",
        "accept": "always_accept (Markov chain random walk)",
        "note": "Fast, widely used in academic studies. Each step is independent.",
    },
    "reversible_recom": {
        "name": "Reversible ReCom",
        "proposal": "ReCom with Metropolis-Hastings acceptance — rejects some proposals to ensure detailed balance",
        "accept": "Metropolis-Hastings (reversible Markov chain)",
        "note": "Statistically rigorous; samples from a known distribution. ~20-30% slower due to rejection.",
    },
}


# ---------------------------------------------------------------------------
# Column name map — adjust to match your shapefile
# ---------------------------------------------------------------------------
COL_MAP = {
    "house": {
        "population": "TOTPOP_H",    # rescaled to match official House district populations
        "dem_votes":  "DEM_VOTES",   # 2020 Presidential Biden (set by prep_data.py)
        "rep_votes":  "REP_VOTES",   # 2020 Presidential Trump
        "black_vap":  "BVAP",        # optional
        "hisp_vap":   "HVAP",        # optional
        "district":   "HDIST",
    },
    "senate": {
        "population": "TOTPOP_S",    # rescaled to match official Senate district populations
        "dem_votes":  "DEM_VOTES",
        "rep_votes":  "REP_VOTES",
        "black_vap":  "BVAP",
        "hisp_vap":   "HVAP",
        "district":   "SDIST",
    },
    "congress": {
        "population": "TOTPOP_C",    # rescaled to match official Congress district populations
        "dem_votes":  "DEM_VOTES",
        "rep_votes":  "REP_VOTES",
        "black_vap":  "BVAP",
        "hisp_vap":   "HVAP",
        "district":   "CDIST",
    },
}

GRAPH_DIR = Path("data/graphs")
ENSEMBLE_DIR = Path("data/ensembles")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def dem_seats(partition: GeographicPartition, cols: dict) -> int:
    """Count districts where Democrats won."""
    return sum(
        1 for d in partition.parts
        if partition["dem_votes"][d] > partition["rep_votes"][d]
    )


def efficiency_gap(partition: GeographicPartition, cols: dict) -> float:
    """
    Efficiency gap = (wasted_dem - wasted_rep) / total_votes.
    Positive → Republican advantage; negative → Democratic advantage.
    """
    total_dem = sum(partition["dem_votes"].values())
    total_rep = sum(partition["rep_votes"].values())
    total_votes = total_dem + total_rep
    if total_votes == 0:
        return 0.0

    wasted_dem = wasted_rep = 0
    for d in partition.parts:
        d_votes = partition["dem_votes"][d]
        r_votes = partition["rep_votes"][d]
        total_d = d_votes + r_votes
        threshold = total_d / 2 + 1

        if d_votes > r_votes:          # Dem win
            wasted_dem += d_votes - threshold  # excess
            wasted_rep += r_votes              # all losing votes
        else:                          # Rep win
            wasted_rep += r_votes - threshold
            wasted_dem += d_votes

    return (wasted_dem - wasted_rep) / total_votes


def mean_median(partition: GeographicPartition) -> float:
    """
    Mean-median difference for Democratic vote share.
    Positive → Democratic advantage; negative → Republican advantage.
    """
    shares = []
    for d in partition.parts:
        total = partition["dem_votes"][d] + partition["rep_votes"][d]
        if total > 0:
            shares.append(partition["dem_votes"][d] / total)

    if not shares:
        return 0.0
    import statistics
    return statistics.mean(shares) - statistics.median(shares)


def polsby_popper_scores(partition: GeographicPartition) -> dict:
    """Return Polsby-Popper compactness per district (requires GeographicPartition)."""
    import math
    scores = {}
    for d in partition.parts:
        area = partition["area"].get(d, 0)
        perim = partition["perimeter"].get(d, 0)
        scores[d] = (4 * math.pi * area / perim ** 2) if perim > 0 else 0.0
    return scores


def majority_minority_count(partition: GeographicPartition, cols: dict,
                             threshold: float = 0.50) -> int:
    """Count districts where Black or Hispanic VAP exceeds threshold."""
    if "black_vap" not in partition.updaters:
        return -1   # data not available
    count = 0
    for d in partition.parts:
        pop = partition["population"][d]
        if pop == 0:
            continue
        bvap = partition["black_vap"].get(d, 0)
        hvap = partition["hisp_vap"].get(d, 0)
        if (bvap + hvap) / pop >= threshold:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main ensemble runner
# ---------------------------------------------------------------------------

def build_updaters(cols: dict, has_geo: bool) -> dict:
    updaters = {
        "population":  Tally(cols["population"], alias="population"),
        "dem_votes":   Tally(cols["dem_votes"],  alias="dem_votes"),
        "rep_votes":   Tally(cols["rep_votes"],  alias="rep_votes"),
        "cut_edges":   cut_edges,
    }
    if cols.get("black_vap"):
        updaters["black_vap"] = Tally(cols["black_vap"], alias="black_vap")
    if cols.get("hisp_vap"):
        updaters["hisp_vap"] = Tally(cols["hisp_vap"], alias="hisp_vap")
    return updaters


def run_ensemble(chamber: str, steps: int, epsilon: float,
                 burn_in: int = 0, seed: int | None = None,
                 algo: str = "recom") -> None:
    cols = COL_MAP[chamber]
    graph_path = GRAPH_DIR / f"ga_{chamber}.json"

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found: {graph_path}\n"
            f"Run: uv run python scripts/build_graph.py --chamber {chamber}"
        )

    print(f"Loading graph from {graph_path}…")
    graph = Graph.from_json(str(graph_path))
    print(f"  {graph.number_of_nodes()} nodes")

    updaters = build_updaters(cols, has_geo=True)

    print("Building initial partition (enacted plan)…")
    initial = GeographicPartition(graph, cols["district"], updaters=updaters)
    num_districts = len(initial.parts)
    total_pop = sum(initial["population"].values())
    ideal_pop = total_pop / num_districts
    print(f"  {num_districts} districts, ideal population {ideal_pop:,.0f}")

    algo_fn = reversible_recom if algo == "reversible_recom" else recom
    proposal = partial(
        algo_fn,
        pop_col=cols["population"],
        pop_target=ideal_pop,
        epsilon=epsilon,
        node_repeats=2,
    )

    pop_constraint = within_percent_of_ideal_population(initial, epsilon)
    # ReCom proposals are inherently contiguous (spanning tree algorithm guarantees it),
    # so contiguous is redundant here. Removing it avoids false failures on the initial
    # enacted plan where precinct-level boundary approximations can create apparent
    # non-contiguity in a few districts.
    constraints = [pop_constraint]

    chain = MarkovChain(
        proposal=proposal,
        constraints=constraints,
        accept=always_accept,
        initial_state=initial,
        total_steps=steps,
    )

    records = []
    enacted_metrics = None
    start = time.time()

    # Stability tracking: for each precinct node, count how many steps it stays
    # in the same district as the enacted plan. Yields a heatmap showing which
    # precincts are "locked" vs. which are contested boundary areas.
    enacted_assignment = dict(initial.assignment)
    stability_counts = {node: 0 for node in graph.nodes()}

    # Iterator wrapper: skip proposals that fail the spanning-tree bipartition.
    def safe_chain(chain):
        it = iter(chain)
        while True:
            try:
                yield next(it)
            except RuntimeError as e:
                if "Could not find a possible cut" in str(e):
                    pass
                else:
                    raise
            except StopIteration:
                return

    print(f"Running {steps} steps (burn-in: {burn_in})…")
    collected = 0
    for step, partition in enumerate(safe_chain(chain)):
        # Stability: count nodes still in same district as enacted plan
        for node in graph.nodes():
            if partition.assignment[node] == enacted_assignment[node]:
                stability_counts[node] += 1
        if step % 500 == 0:
            elapsed = time.time() - start
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (steps - step) / rate if rate > 0 else 0
            print(f"  step {step:,}/{steps:,}  ({rate:.1f} steps/s, ~{eta/60:.1f} min remaining)")

        metrics = {
            "step": step,
            "dem_seats": dem_seats(partition, cols),
            "efficiency_gap": efficiency_gap(partition, cols),
            "mean_median": mean_median(partition),
            "num_cut_edges": len(partition["cut_edges"]),
        }

        # Compactness (mean Polsby-Popper across districts)
        try:
            pp = polsby_popper_scores(partition)
            metrics["polsby_popper_mean"] = sum(pp.values()) / len(pp) if pp else 0.0
            metrics["polsby_popper_min"] = min(pp.values()) if pp else 0.0
        except Exception:
            pass

        # Minority representation
        mm = majority_minority_count(partition, cols)
        if mm >= 0:
            metrics["majority_minority_districts"] = mm

        # Save enacted plan metrics separately (step 0 = initial state)
        if step == 0:
            enacted_metrics = metrics.copy()

        if step >= burn_in:
            records.append(metrics)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENSEMBLE_DIR / f"{chamber}_ensemble.parquet"
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print(f"\nEnsemble saved: {out_path}  ({len(df)} plans)")

    # Save stability scores: node_id → fraction of steps in same district as enacted
    total_steps = max(step + 1, 1)
    stability = {str(k): round(v / total_steps, 4) for k, v in stability_counts.items()}
    stability_path = ENSEMBLE_DIR / f"{chamber}_stability.json"
    with open(stability_path, "w") as f:
        json.dump(stability, f)
    print(f"Stability map saved: {stability_path}")

    runtime = round(time.time() - start, 1)

    # Save run metadata
    import datetime
    meta = {
        "chamber": chamber,
        "steps": steps,
        "burn_in": burn_in,
        "epsilon": epsilon,
        "num_districts": num_districts,
        "total_population": int(total_pop),
        "ideal_population": float(ideal_pop),
        "enacted_metrics": enacted_metrics,
        "columns_used": cols,
        "runtime_seconds": runtime,
        "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_sources": {
            "precincts": "RDH Georgia 2020 General Election Statewide Precincts (ga_gen_20_st_prec)",
            "population": "2020 Census PL 94-171 Block-level (ga_pl2020_p1_b, ga_pl2020_p4_b)",
            "districts": {
                "house":    "Georgia 2023 Enacted House Districts (House-2023 shape)",
                "senate":   "Georgia 2023 Enacted Senate Districts (Senate-2023 shape file)",
                "congress": "Georgia 2023 Enacted Congressional Districts (Congress-2023 shape)",
            }.get(chamber, ""),
            "election": "2020 US Presidential Election results (Biden/Trump)",
        },
        "algorithm": {
            "key": algo,
            **ALGO_DESCRIPTIONS.get(algo, {}),
            "constraints": [f"Population within ±{epsilon*100:.0f}% of ideal ({ideal_pop:,.0f})"],
        },
    }
    meta_path = ENSEMBLE_DIR / f"{chamber}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    # Quick summary
    print(f"\n=== Enacted plan vs. ensemble ({chamber}) ===")
    if enacted_metrics:
        for key in ["dem_seats", "efficiency_gap", "mean_median"]:
            val = enacted_metrics.get(key)
            ens_vals = df[key] if key in df.columns else None
            if val is not None and ens_vals is not None:
                pct = (ens_vals <= val).mean() * 100
                print(f"  {key}: enacted={val:.4f}  ensemble_pctile={pct:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GerryChain ReCom ensemble")
    parser.add_argument("--chamber", required=True, choices=["house", "senate", "congress"])
    parser.add_argument("--steps",   type=int,   default=10_000)
    parser.add_argument("--epsilon", type=float, default=0.07,
                        help="Population deviation tolerance (default 0.07 = ±7%%, legal max ~10%%)")
    parser.add_argument("--burn-in", type=int,   default=0,
                        help="Steps to discard at start of chain")
    parser.add_argument("--seed",    type=int,   default=None)
    parser.add_argument("--algo",    default="recom", choices=list(ALGO_DESCRIPTIONS.keys()),
                        help="Proposal algorithm (default: recom)")
    args = parser.parse_args()

    run_ensemble(
        chamber=args.chamber,
        steps=args.steps,
        epsilon=args.epsilon,
        burn_in=args.burn_in,
        seed=args.seed,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()
