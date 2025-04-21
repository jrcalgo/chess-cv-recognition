import os
from io import BytesIO

import cairosvg
from PIL import Image
import numpy as np


def _load_svg_as_array(path: str, size: int) -> np.ndarray:
    png_data = cairosvg.svg2png(
        url=path,
        output_width=size,
        output_height=size
    )
    buf = BytesIO(png_data)
    img = Image.open(buf).convert("RGBA")
    return np.array(img)


def parse_sprites(scale_size: int = 32):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(this_dir)

    assets_dir = os.path.join(project_dir, "assets")

    piece_map = {
        "bishop": "B",
        "knight": "Kn",
        "king":   "K",
        "queen":  "Q",
        "rook":   "R",
        "pawn":   "P",
    }

    white_dir = os.path.join(assets_dir, "white_pieces")
    black_dir = os.path.join(assets_dir, "black_pieces")

    white_pieces = {}
    black_pieces = {}

    for prefix, directory, out_dict in [
        ("w", white_dir, white_pieces),
        ("b", black_dir, black_pieces),
    ]:
        for fname in os.listdir(directory):
            if not fname.lower().endswith(".svg"):
                continue
            # e.g. "white_bishop.svg" or "black_knight.svg", etc.
            base = os.path.splitext(fname)[0]
            _, piece_name = base.split("_", 1)
            code = piece_map.get(piece_name)
            if not code:
                continue

            key = f"{prefix}{code}"
            full_path = os.path.join(directory, fname)
            out_dict[key] = _load_svg_as_array(full_path, scale_size)

    return white_pieces, black_pieces
