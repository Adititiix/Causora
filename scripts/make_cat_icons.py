#!/usr/bin/env python3
"""Make the chat icons: the original cat drawings (assistant) and frog (user), with a
transparent background.

The originals (src/incident_copilot/ui/assets/originals/) are pixel art on a white
background, and the cats themselves are white or cream inside a black outline. So "make
every near-white pixel transparent" would punch holes in the cat. Instead:

  1. Flood-fill the background from the image border, 4-connected, over near-white pixels
     only. A 4-connected fill cannot pass the diagonal steps of a pixel-art outline, so it
     stops at the outline and never reaches the cat's fill.
  2. Seed only where the border really is background. The thinking cat is cropped at the
     bottom, so its body touches the bottom edge; there the fill is seeded only from each
     corner inward, up to the first pixel that is not near-white.
  3. Clear the light, unsaturated pixels left right next to the background (resampling
     noise along the outline), so there is no white halo on a dark page.
  4. Crop to the cat, pad to a square, keep hard pixel edges (no blur; any resize is
     nearest-neighbour).

The originals are only read. Re-run after replacing them:

    python scripts/make_cat_icons.py
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "src" / "incident_copilot" / "ui" / "assets"
# original file -> icon file
ICONS = {"cat-avatar-open.png": "cat_avatar.png", "cat-thinking.png": "cat_thinking.png",
         # The user's avatar (a JPEG).
         "frog_user.jpg": "frog_avatar.png"}

BG_MIN = 236        # background: every channel at least this ...
BG_SPREAD = 14      # ... and channels within this of each other (white, not cream or pink)
HALO_MIN = 170      # outline-side noise: light and unsaturated, next to the background
HALO_SPREAD = 24
HALO_PASSES = 2
PAD = 4             # transparent pixels around the cropped cat


def _near(px: tuple[int, int, int], low: int, spread: int) -> bool:
    return min(px) >= low and max(px) - min(px) <= spread


def background_mask(img: Image.Image) -> list[list[bool]]:
    """True for background pixels: near-white and 4-connected to a true-background seed."""
    w, h = img.size
    px = img.load()
    is_bg = [[_near(px[x, y], BG_MIN, BG_SPREAD) for x in range(w)] for y in range(h)]
    seeds = [(x, 0) for x in range(w)] + [(0, y) for y in range(h)] \
        + [(w - 1, y) for y in range(h)]
    # Bottom edge: only the runs that start at a corner (the cat may be cropped there).
    for xs in (range(w), range(w - 1, -1, -1)):
        for x in xs:
            if not is_bg[h - 1][x]:
                break
            seeds.append((x, h - 1))
    mask = [[False] * w for _ in range(h)]
    queue = deque((x, y) for x, y in seeds if is_bg[y][x])
    for x, y in queue:
        mask[y][x] = True
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny][nx] and is_bg[ny][nx]:
                mask[ny][nx] = True
                queue.append((nx, ny))
    return mask


def remove_halo(img: Image.Image, mask: list[list[bool]]) -> int:
    """Add light, unsaturated pixels touching the background to it. Returns how many."""
    w, h = img.size
    px = img.load()
    added = 0
    for _ in range(HALO_PASSES):
        ring = [(x, y) for y in range(h) for x in range(w)
                if not mask[y][x] and _near(px[x, y], HALO_MIN, HALO_SPREAD)
                and any(0 <= nx < w and 0 <= ny < h and mask[ny][nx]
                        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))]
        for x, y in ring:
            mask[y][x] = True
        added += len(ring)
        if not ring:
            break
    return added


def make_icon(src: Path, dst: Path, size: int | None = None) -> str:
    original = Image.open(src)
    info = f"{src.name}: {original.size[0]}x{original.size[1]} {original.mode}"
    rgb = original.convert("RGB")
    mask = background_mask(rgb)
    halo = remove_halo(rgb, mask)
    out = rgb.convert("RGBA")
    w, h = out.size
    alpha = Image.new("L", (w, h))
    alpha.putdata([0 if mask[y][x] else 255 for y in range(h) for x in range(w)])
    out.putalpha(alpha)
    box = alpha.getbbox()
    if box is None:
        raise SystemExit(f"{src.name}: nothing left after removing the background")
    cat = out.crop(box)
    side = max(cat.size) + 2 * PAD
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    # Centred horizontally; a cat cropped at the bottom stays on the bottom edge.
    top = PAD if box[3] < h else side - cat.size[1]
    square.paste(cat, ((side - cat.size[0]) // 2, top))
    if size:
        square = square.resize((size, size), Image.Resampling.NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    square.save(dst, optimize=True)
    transparent = sum(row.count(True) for row in mask)
    return (f"{info} -> {dst.name}: {square.size[0]}x{square.size[1]} RGBA, "
            f"{transparent} background px removed ({halo} of them outline noise)")


BLOCK = 16  # the cat's pixel-art grid in cat_avatar.png (no resize)
# A small closed smile, in blocks: row 0 and row 1 of a 5-block-wide pattern.
CLOSED_MOUTH = ("X...X", ".XXX.")


def _is_mouth(px) -> bool:
    r, g, b, a = px
    return a > 0 and r > 200 and g < 160  # the open mouth's reds and pinks, not the pale ears


def make_closed_mouth(open_icon: Path, dst: Path) -> str:
    """The speaking animation's second frame: the open-mouth cat with its mouth painted over in
    the face's white and a small closed mouth drawn on the same pixel grid, under the nose.
    Derived from cat_avatar.png, so the two frames line up exactly."""
    im = Image.open(open_icon).convert("RGBA")
    w, h = im.size
    px = im.load()
    mouth = [(x, y) for y in range(h) for x in range(w) if _is_mouth(px[x, y])]
    # Two red/pink regions: the nose on top, then (after rows without any) the open mouth.
    rows = sorted({y for _, y in mouth})
    gap = next((a for a, b in zip(rows, rows[1:], strict=False) if b > a + 1), None)
    if gap is None:
        raise SystemExit("no open mouth found below the nose in the cat icon")
    nose = [(x, y) for x, y in mouth if y <= gap]
    body = [(x, y) for x, y in mouth if y > gap]
    x0, x1 = min(x for x, _ in body), max(x for x, _ in body)
    y1 = max(y for _, y in body)
    nose_x0, nose_x1 = min(x for x, _ in nose), max(x for x, _ in nose)
    nose_y1 = max(y for _, y in nose)

    def near_transparent(x, y) -> bool:  # the cat's own outline stays untouched
        return any(px[i, j][3] == 0 for i in range(max(0, x - BLOCK), min(w, x + BLOCK + 1))
                   for j in range((max(0, y - BLOCK)), min(h, y + BLOCK + 1)))

    white = (255, 255, 255, 255)
    # Below the nose only. The mouth, its outline and the original's 1-px anti-aliasing lines
    # (dark, or light grey: not pure white) become face white; the outer outline is kept.
    def is_mark(c) -> bool:
        grey = max(c[:3]) - min(c[:3]) < 40 and c[:3] != (255, 255, 255)
        return _is_mouth(c) or max(c[:3]) < 200 or grey

    for y in range(nose_y1 + 1, min(h, y1 + BLOCK + 1)):
        for x in range(max(0, x0 - BLOCK), min(w, x1 + BLOCK + 1)):
            c = px[x, y]
            if c[3] and is_mark(c) and not near_transparent(x, y):
                px[x, y] = white
    # Snap to the grid the nose sits on, centred under it, one block below it.
    centre = (nose_x0 + nose_x1 + 1) // 2
    left = nose_x0 + ((centre - len(CLOSED_MOUTH[0]) * BLOCK // 2) - nose_x0) // BLOCK * BLOCK
    top = nose_y1 + 1 + BLOCK
    for r, row in enumerate(CLOSED_MOUTH):
        for c, cell in enumerate(row):
            if cell == "X":
                for y in range(top + r * BLOCK, top + (r + 1) * BLOCK):
                    for x in range(left + c * BLOCK, left + (c + 1) * BLOCK):
                        px[x, y] = (0, 0, 0, 255)
    im.save(dst, optimize=True)
    return f"{open_icon.name} -> {dst.name}: closed-mouth frame, {w}x{h} RGBA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=None,
                        help="optional square output size (nearest-neighbour)")
    args = parser.parse_args()
    for name, icon in ICONS.items():
        src = ASSETS / "originals" / name
        if not src.exists():
            raise SystemExit(f"missing original: {src}")
        print(make_icon(src, ASSETS / icon, args.size))
    if not args.size:  # the closed mouth is drawn on the unscaled 16 px grid
        print(make_closed_mouth(ASSETS / "cat_avatar.png", ASSETS / "cat_avatar_closed.png"))
        print(make_sprite(ASSETS / "cat_avatar.png", ASSETS / "cat_avatar_closed.png",
                          ASSETS / "cat_speaking.png"))


def make_sprite(open_frame: Path, closed_frame: Path, dst: Path) -> str:
    """Both frames side by side (open | closed): the speaking avatar shows one half at a time
    (CSS object-position), so the two frames can never drift apart."""
    a, b = Image.open(open_frame).convert("RGBA"), Image.open(closed_frame).convert("RGBA")
    if a.size != b.size:
        raise SystemExit("the open and closed frames differ in size")
    sheet = Image.new("RGBA", (a.width * 2, a.height), (0, 0, 0, 0))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (a.width, 0))
    sheet.save(dst, optimize=True)
    return f"{dst.name}: {sheet.width}x{sheet.height} sprite (open | closed)"


if __name__ == "__main__":
    main()
