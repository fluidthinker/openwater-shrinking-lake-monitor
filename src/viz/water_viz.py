from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_figure_png(fig: plt.Figure, out_png: Path, dpi: int = 200) -> None:
    """Save a matplotlib figure to a PNG path."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dpi=dpi, fname=out_png, bbox_inches="tight", facecolor=fig.get_facecolor())

from matplotlib.colors import ListedColormap

def make_watermask_overlay_figure(
    water,
    aoi_gdf,
    crs: str,
    title: str,
    *,
    background_color: str = "white",
    water_color: str = "#1E63FF",   # nice blue
    water_alpha: float = 0.85,
    aoi_color: str = "orange",
    aoi_linewidth: float = 2.0,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a georeferenced overlay: water mask + AOI boundary.

    IMPORTANT: uses xarray plotting (georeferenced) so AOI aligns correctly.
    """
    aoi_proj = aoi_gdf.to_crs(crs)

    fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    ax.set_facecolor(background_color)

    # Make False pixels transparent by turning them into NaN
    water_show = water.where(water).astype("float32")

    # Single-color colormap for water; NaNs won't be drawn
    cmap = ListedColormap([water_color])

    water_show.plot(
        ax=ax,
        add_colorbar=False,
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=water_alpha,
    )

    aoi_proj.boundary.plot(ax=ax, linewidth=aoi_linewidth, color=aoi_color)

    ax.set_title(title)
    ax.set_axis_off()
    return fig, ax



@dataclass(frozen=True)
class StoryStyle:
    """Style settings for mask-only story frames."""
    figsize: Tuple[float, float] = (8, 8)
    dpi: int = 200

    # Canvas
    background_color: str = "#E8D8B5"  # sand-ish

    # Water color
    water_color_rgb: Tuple[float, float, float] = (0.10, 0.35, 0.90)
    water_alpha: float = 0.90

    # Reserve whitespace so labels don't cover the lake
    margin_left: float = 0.28   # fraction of figure width reserved on the left
    margin_right: float = 0.02
    margin_top: float = 0.02
    margin_bottom: float = 0.02

    # Reference outline
    ref_outline_color: str = "#444444"  # dark gray
    ref_outline_width: float = 2.0


    # Label
    show_label: bool = True
    label_fontsize: int = 18
    label_color: str = "black"
    label_ha: str = "left"
    label_va: str = "top"

    # Label box
    label_box: bool = True
    label_box_facecolor: str = "white"
    label_box_alpha: float = 0.70

def render_story_frame(
    water,
    label: Optional[str],
    *,
    style: StoryStyle = StoryStyle(),
    reference_water=None,  # NEW: DataArray boolean mask from START_YEAR
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a mask-only story frame with optional reference-outline overlay."""
    w = np.asarray(water.values).astype(bool)

    rgba = np.zeros((w.shape[0], w.shape[1], 4), dtype=np.float32)
    r, g, b = style.water_color_rgb
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = w.astype(np.float32) * float(style.water_alpha)

    fig, ax = plt.subplots(figsize=style.figsize, facecolor=style.background_color)
    ax.set_facecolor(style.background_color)

    # Reserve margin (your existing margin solution)
    left = style.margin_left
    bottom = style.margin_bottom
    width = 1.0 - style.margin_left - style.margin_right
    height = 1.0 - style.margin_top - style.margin_bottom
    ax.set_position([left, bottom, width, height])

    ax.imshow(rgba, origin="upper")

    # --- Reference outline (drawn once, reused) ---
    if reference_water is not None:
        ref = np.asarray(reference_water.values).astype(bool)
        if ref.shape == w.shape:
            # Draw boundary of reference mask as a contour line
            ax.contour(
                ref.astype(np.uint8),
                levels=[0.5],
                colors=[style.ref_outline_color],
                linewidths=style.ref_outline_width,
            )
        else:
            print(
                f"⚠️ Reference outline skipped (shape mismatch): "
                f"ref={ref.shape}, current={w.shape}"
            )

    ax.set_axis_off()

    # Label in reserved margin (figure coordinates)
    if style.show_label and label:
        bbox = None
        if style.label_box:
            bbox = dict(
                boxstyle="round,pad=0.35",
                facecolor=style.label_box_facecolor,
                alpha=style.label_box_alpha,
                edgecolor="none",
            )

        fig.text(
            0.02, 0.98,
            label,
            ha=style.label_ha,
            va=style.label_va,
            fontsize=style.label_fontsize,
            color=style.label_color,
            bbox=bbox,
        )

    return fig, ax





def plot_month_diagnostics(
    metrics: Dict[str, Any],
    cfg,
    out_dir: Optional[Path] = None,
    *,
    save: bool = True,
    show: bool = True,
) -> None:
    """Plot and optionally save diagnostic figures for a single month.

    Expects metrics from compute_month_metrics() to include:
    - valid_fraction_map
    - ndwi_med
    - water
    - aoi
    """
    if metrics.get("water") is None:
        print("No data for this month; skipping plots.")
        return

    year = int(metrics["year"])
    month = int(metrics["month"])

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Valid fraction map
    fig, ax = plt.subplots()
    metrics["valid_fraction_map"].plot(ax=ax, robust=True)
    ax.set_title(f"Valid (clear) observation fraction — {year}-{month:02d}")
    ax.set_axis_off()
    if save and out_dir is not None:
        save_figure_png(fig, out_dir / f"valid_fraction_{year}-{month:02d}.png")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 2) NDWI median composite
    fig, ax = plt.subplots()
    metrics["ndwi_med"].plot(ax=ax, robust=True)
    ax.set_title(f"NDWI median composite — {year}-{month:02d}")
    ax.set_axis_off()
    if save and out_dir is not None:
        save_figure_png(fig, out_dir / f"ndwi_median_{year}-{month:02d}.png")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 3) Water mask + AOI overlay (QA figure)
    title = f"Water mask vs AOI outline — {year}-{month:02d} (NDWI>{cfg.water_thresh})"
    fig, _ = make_watermask_overlay_figure(
        metrics["water"],
        metrics["aoi"],
        cfg.load_crs,
        title,
        water_alpha=0.85,
        aoi_color="orange",
        aoi_linewidth=2.0,
        background_color="white",
    )
    if save and out_dir is not None:
        save_figure_png(
            fig,
            out_dir / f"overlay_watermask_vs_aoi_{year}-{month:02d}_ndwi{cfg.water_thresh:.2f}.png",
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
