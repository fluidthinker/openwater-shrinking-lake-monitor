# %% [markdown]
# # 02 — Monthly loop (2016 → present)
#
# Goal:
# - Run the validated one-month NDWI workflow across a range of months
# - Save a tidy time series CSV:
#     year, month, water_area_km2, median_valid_fraction, valid_fraction_any, n_items
# - Support resume: skip months already present in the CSV
# - Optional: save diagnostic figures for selected months only (e.g., Jan/Jul each year)

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

# Ensure repo root is on sys.path (Option A)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---- IMPORT YOUR COMPUTE + PLOT HELPERS ----
# Recommended: move your helpers into src/ (e.g., src/ndwi/monthly.py) and import from there.
#
# Example (recommended):
# from src.ndwi.monthly import NdwiConfig, read_ndwi_config, compute_month_metrics, plot_month_diagnostics
#
# If you haven't moved them yet and they're still inside scripts/01_one_month_smoketest.py,
# you *can* import from scripts, but that's not ideal long-term.
# Example (temporary):
# from scripts._01_one_month_smoketest import NdwiConfig, read_ndwi_config, compute_month_metrics, plot_month_diagnostics

from src.ndwi.monthly import (  # <-- YOU WILL CREATE THIS MODULE OR ADJUST IMPORTS
    NdwiConfig,
    read_ndwi_config,
    compute_month_metrics,
    plot_month_diagnostics,
)

# %% [markdown]
# ## User knobs

# %%
# Date range (inclusive)
START_YEAR = 2016
START_MONTH = 1

END_YEAR = 2016
END_MONTH = 12

# Input paths
AOI_GEOJSON = REPO_ROOT / "data" / "external" / "aoi.geojson"
CFG_NDWI = REPO_ROOT / "configs" / "ndwi.yaml"

# Output paths
OUT_TABLES = REPO_ROOT / "outputs" / "tables"
OUT_FIGURES = REPO_ROOT / "outputs" / "figures"
OUT_CSV = OUT_TABLES / "water_area_timeseries.csv"

# Resume behavior
RESUME_IF_EXISTS = True

# Optional plotting:
# - None: no plots saved
# - "selected": only save plots for months listed in PLOT_MONTHS (e.g., [1, 7])
# - "all": save plots for every processed month (not recommended unless you really want it)
PLOT_MODE: str = "selected"  # "none" | "selected" | "all"
PLOT_MONTHS: List[int] = [1]  # used when PLOT_MODE == "selected"


# %% [markdown]
# ## Helpers

# %%
def iter_months(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> Iterable[Tuple[int, int]]:
    """Yield (year, month) pairs from start to end inclusive.
    produces a clean sequence like: (2020,1) (2020,2) ... (2020,12)
    
    """
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def load_existing_results(path: Path) -> pd.DataFrame:
    """Load existing results CSV if it exists, else return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "year",
                "month",
                "water_area_km2",
                "median_valid_fraction",
                "valid_fraction_any",
                "n_items",
            ]
        )
    df = pd.read_csv(path)
    # Normalize dtypes
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def month_key(year: int, month: int) -> str:
    """Return YYYY-MM string key."""
    return f"{year:04d}-{month:02d}"


def should_plot_month(year: int, month: int) -> bool:
    """Decide whether to plot/save diagnostics for a given month."""
    if PLOT_MODE.lower() == "none":
        return False
    if PLOT_MODE.lower() == "all":
        return True
    # selected
    return month in set(PLOT_MONTHS)


# %% [markdown]
# ## Run loop

# %%
def main() -> None:
    """Run monthly loop and write results CSV."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    cfg = read_ndwi_config(CFG_NDWI)

    print(f"Repo root: {REPO_ROOT}")
    print(f"AOI: {AOI_GEOJSON}")
    print(f"Config: {CFG_NDWI} (exists={CFG_NDWI.exists()})")
    print(
        "Settings: "
        f"cloud_cover_max={cfg.cloud_cover_max}, "
        f"water_thresh={cfg.water_thresh}, "
        f"resolution_m={cfg.resolution_m}, "
        f"crs={cfg.load_crs}"
    )
    print(f"Output CSV: {OUT_CSV}")
    print("")

    existing = load_existing_results(OUT_CSV)
    done = set(month_key(y, m) for y, m in zip(existing["year"], existing["month"]))

    rows: List[Dict[str, Any]] = []
    n_total = 0
    n_skipped = 0
    n_ran = 0

    for year, month in iter_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        n_total += 1
        key = month_key(year, month)

        if RESUME_IF_EXISTS and key in done:
            n_skipped += 1
            continue

        print(f"Processing {key} ...")

        try:
            metrics = compute_month_metrics(year, month, AOI_GEOJSON, cfg=cfg)
            rows.append(
                {
                    "year": int(metrics["year"]),
                    "month": int(metrics["month"]),
                    "water_area_km2": metrics["water_area_km2"],
                    "median_valid_fraction": metrics["median_valid_fraction"],
                    "valid_fraction_any": metrics["valid_fraction_any"],
                    "n_items": int(metrics["n_items"]),
                }
            )

            # Optional diagnostics (saved to outputs/figures/)
            if should_plot_month(year, month):
                plot_month_diagnostics(metrics, cfg, out_dir=OUT_FIGURES)

            n_ran += 1

            # Write incrementally so you can stop/restart safely
            df_new = pd.DataFrame(rows)
            df_all = pd.concat([existing, df_new], ignore_index=True)

            df_all = (
                df_all.drop_duplicates(subset=["year", "month"], keep="last")
                .sort_values(["year", "month"])
                .reset_index(drop=True)
            )
            df_all.to_csv(OUT_CSV, index=False)

        except Exception as e:
            # Fail-soft: record NaNs for this month and continue
            print(f"  ⚠️ Failed {key}: {type(e).__name__}: {e}")

            rows.append(
                {
                    "year": year,
                    "month": month,
                    "water_area_km2": float("nan"),
                    "median_valid_fraction": float("nan"),
                    "valid_fraction_any": float("nan"),
                    "n_items": 0,
                }
            )

            df_new = pd.DataFrame(rows)
            df_all = pd.concat([existing, df_new], ignore_index=True)
            df_all = (
                df_all.drop_duplicates(subset=["year", "month"], keep="last")
                .sort_values(["year", "month"])
                .reset_index(drop=True)
            )
            df_all.to_csv(OUT_CSV, index=False)

    print("")
    print("Done.")
    print(f"Months in range: {n_total}")
    print(f"Skipped (already present): {n_skipped}")
    print(f"Processed this run: {n_ran}")
    print(f"Wrote: {OUT_CSV}")


# %%
if __name__ == "__main__":
    main()
