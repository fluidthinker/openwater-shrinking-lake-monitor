
# %% [markdown]
# ------------------------------------------------------------------
# Imports and data loading
# ------------------------------------------------------------------


# %% 
from pathlib import Path
import pandas as pd

# Resolve repo root from this file
REPO_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = REPO_ROOT / "outputs" / "tables" / "water_area_timeseries.csv"

df = pd.read_csv(CSV_PATH)

df.head()


# %% [markdown]
# ------------------------------------------------------------------
# Basic cleanup and type enforcement
# ------------------------------------------------------------------

# %% 
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

df.describe()


# %% [markdown]
# ------------------------------------------------------------------
# A. Annual minimum water area
# "Worst water condition reached each year"
# ------------------------------------------------------------------

# %%
annual_min = (
    df.groupby("year")["water_area_km2"]
    .min()
    .rename("annual_min_km2")
)

annual_min


# %% [markdown]
# ------------------------------------------------------------------
# B. Late-season average (Aug–Oct)
# "Typical dry-season condition"
# ------------------------------------------------------------------
# %%
late_season = df[df["month"].isin([8, 9, 10])]

late_season_avg = (
    late_season.groupby("year")["water_area_km2"]
    .mean()
    .rename("late_season_avg_km2")
)

late_season_avg




# %% [markdown]
# ------------------------------------------------------------------
# C. Seasonal integral (May–Oct)
# Two forms:
#   - Mean: average condition across season
#   - Sum: total water availability over season
# ------------------------------------------------------------------
# %%
seasonal = df[df["month"].between(5, 10)]

seasonal_mean = (
    seasonal.groupby("year")["water_area_km2"]
    .mean()
    .rename("seasonal_mean_km2")
)

seasonal_sum = (
    seasonal.groupby("year")["water_area_km2"]
    .sum()
    .rename("seasonal_sum_km2_months")
)

seasonal_mean, seasonal_sum


# %% [markdown]
# ------------------------------------------------------------------
# D. 10th percentile by year
# "Robust low-water indicator (less fragile than min)"
# ------------------------------------------------------------------
# %%
p10 = (
    df.groupby("year")["water_area_km2"]
    .quantile(0.10)
    .rename("p10_km2")
)

p10


# %% [markdown]
# ------------------------------------------------------------------
# Combine all metrics into a single summary table
# ------------------------------------------------------------------

# %%
summary = pd.concat(
    [
        annual_min,
        late_season_avg,
        seasonal_mean,
        seasonal_sum,
        p10,
    ],
    axis=1,
)

summary = summary.round(2)

print(f"summary {summary}")


# %% [markdown]
# ------------------------------------------------------------------
# (Optional) Save summary table for plotting / reporting
# ------------------------------------------------------------------

# %%
OUTPUT_PATH = REPO_ROOT / "outputs" / "tables" / "water_area_summary_metrics.csv"
summary.to_csv(OUTPUT_PATH)

print(f"Saved summary metrics to: {OUTPUT_PATH}")

# %%
