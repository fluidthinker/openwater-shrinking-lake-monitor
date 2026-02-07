# %%
# ------------------------------------------------------------------
# Create a September late-season GIF animation from story frames
#
# This script:
# - Collects September water-mask PNGs (one per year)
# - Sorts them chronologically by year
# - Assembles them into a GIF with controlled frame timing
#
# Output:
# outputs/maps/story_sept_2019_2025_2000ms.gif
# ------------------------------------------------------------------

from pathlib import Path
import re
from PIL import Image

# ------------------------------------------------------------------
# Resolve repository paths
# ------------------------------------------------------------------

# Determine repository root based on this script's location
REPO_ROOT = Path(__file__).resolve().parents[1]

# Directory containing September story frames (PNG images)
FRAMES_DIR = REPO_ROOT / "outputs" / "maps" / "story_frames_09"

# Output directory for the final animation
OUTPUT_DIR = REPO_ROOT / "outputs" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output GIF path (include timing in filename to avoid caching confusion)
out_gif = OUTPUT_DIR / "story_sept_2019_2025_2000ms.gif"

# ------------------------------------------------------------------
# Helper function: extract year from filename for sorting
# ------------------------------------------------------------------
def sort_key(p: Path) -> int:
    """
    Extract the year from a filename of the form:
    'story_mask_YYYY-09_*.png'

    Returns:
        int: year for chronological sorting; large value if no match
    """
    m = re.search(r"story_mask_(\d{4})-09_", p.name)
    return int(m.group(1)) if m else 9999

# ------------------------------------------------------------------
# Locate and sort September frames
# ------------------------------------------------------------------

# Find all September story-frame PNGs and sort by year
frame_paths = sorted(
    FRAMES_DIR.glob("story_mask_*-09_*.png"),
    key=sort_key,
)

print(f"Frames found: {len(frame_paths)}")

# ------------------------------------------------------------------
# Load images into memory
# ------------------------------------------------------------------

# Open each PNG and convert to RGBA to ensure consistent color handling
frames = [Image.open(p).convert("RGBA") for p in frame_paths]

# ------------------------------------------------------------------
# GIF timing configuration
# ------------------------------------------------------------------

# Pillow expects frame duration in milliseconds
# 2000 ms = 2 seconds per frame (slow, readable playback)
duration_ms = 2000

# ------------------------------------------------------------------
# Convert frames for efficient GIF encoding
# ------------------------------------------------------------------

# Convert RGBA frames to palette (P) mode using an adaptive palette
# This reduces file size and improves compatibility across viewers
frames_p = [
    f.convert("P", palette=Image.Palette.ADAPTIVE)
    for f in frames
]

# ------------------------------------------------------------------
# Save GIF animation
# ------------------------------------------------------------------

# Save the animation:
# - save_all=True enables multi-frame output
# - append_images adds subsequent frames
# - loop=0 means infinite looping
# - optimize=False preserves consistent frame timing
frames_p[0].save(
    out_gif,
    save_all=True,
    append_images=frames_p[1:],
    duration=duration_ms,
    loop=0,
    optimize=False,
)

print(f"Saved GIF animation to:\n{out_gif}")
