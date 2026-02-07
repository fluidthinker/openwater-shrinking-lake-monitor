# %%
from pathlib import Path
import re
import imageio.v2 as imageio


# %%
REPO_ROOT = Path(__file__).resolve().parents[1]
frames_dir = REPO_ROOT / "outputs" / "maps" / "story_frames_09"
out_gif = REPO_ROOT / "outputs" / "maps" / "story_sept_2019_2025.gif"

# %%
# Sort by year from filename "story_mask_YYYY-09_..."
def sort_key(p: Path):
    m = re.search(r"story_mask_(\d{4})-09_", p.name)
    return int(m.group(1)) if m else 9999

# %% 
frame_paths = sorted(frames_dir.glob("story_mask_*-09_*.png"), key=sort_key)

images = [imageio.imread(p) for p in frame_paths]
imageio.mimsave(out_gif, images, duration=0.9, loop=0)  # seconds per frame

print(f"Saved GIF: {out_gif}")

# %%
