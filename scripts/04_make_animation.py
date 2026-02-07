# %% [markdown]
# # September Late-Season Story Animation
#
# This script assembles September water-mask story frames into an animation.
#
# **Purpose**
# - Visually communicate late-season (dry-season) surface-water extent
# - Complement the quantitative late-season (Aug–Oct) average time series
#
# **Design choices**
# - September is used as a representative late-season snapshot
# - Frames are ordered by year
# - Output formats supported:
#   - GIF (easy to embed in README)
#   - MP4 (optional, smoother playback)
#
# Frames are expected to already exist in:
# `outputs/maps/story_frames_09/`
# with filenames like:
# `story_mask_YYYY-09_*.png`

# %%
# ------------------------------------------------------------------
# Imports and path setup
# ------------------------------------------------------------------
from pathlib import Path
import re
import imageio.v2 as imageio

# Resolve repo root from this file location
REPO_ROOT = Path(__file__).resolve().parents[1]

FRAMES_DIR = REPO_ROOT / "outputs" / "maps" / "story_frames_09"
OUTPUT_DIR = REPO_ROOT / "outputs" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Locate and sort September frames
#
# Frames are named using the pattern:
# `story_mask_YYYY-09_*.png`
#
# We extract the year from the filename and sort chronologically so the
# animation progresses forward in time.

# %%
def sort_key(path: Path) -> int:
    """
    Extract year from filename for chronological sorting.
    """
    match = re.search(r"story_mask_(\d{4})-09_", path.name)
    return int(match.group(1)) if match else 9999


frame_paths = sorted(
    FRAMES_DIR.glob("story_mask_*-09_*.png"),
    key=sort_key,
)

print(f"Found {len(frame_paths)} September frames")
for p in frame_paths:
    print(" ", p.name)


# %% [markdown]
# ## Create GIF animation (recommended for README)
#
# - GIFs render reliably on GitHub
# - Frame duration controls playback speed
# - `loop=0` means infinite looping

# %%
out_gif = OUTPUT_DIR / "story_sept_2019_2025.gif"

images = [imageio.imread(p) for p in frame_paths]

imageio.mimsave(
    out_gif,
    images,
    duration=0.9,  # seconds per frame
    loop=0,
)

print(f"Saved GIF animation to:\n{out_gif}")


# %% [markdown]
# ## (Optional) Create MP4 animation
#
# MP4 provides smoother playback and smaller file size, but requires
# `ffmpeg` to be available on the system.
#
# If this fails due to ffmpeg, the GIF above is sufficient for the README.

# %%
out_mp4 = OUTPUT_DIR / "story_sept_2019_2025.mp4"

try:
    with imageio.get_writer(out_mp4, fps=1) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))
    print(f"Saved MP4 animation to:\n{out_mp4}")
except Exception as e:
    print("MP4 creation failed (likely ffmpeg not installed).")
    print("GIF output is still valid.")
    print("Error:", e)


# %% [markdown]
# ## Result
#
# You now have:
#
# - A **late-season quantitative plot** (Aug–Oct average)
# - A **September story animation** showing year-to-year shoreline change
#
# These two outputs are intentionally aligned:
# - The plot summarizes late-season conditions numerically
# - The animation provides an intuitive visual counterpart
#
# Next steps:
# - Embed the GIF in the README
# - Add a short “Visualization strategy” paragraph
# - Stop iterating and ship 🚀



