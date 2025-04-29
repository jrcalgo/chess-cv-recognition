import os
import pygame


def parse_sprites(scale_size: int = 32):
    # Make sure you’ve called pygame.init() before this runs!
    this_dir = os.path.dirname(__file__)
    assets_dir = os.path.join(this_dir, "..", "assets")
    piece_map = {
        "bishop": "B", "knight": "Kn", "king": "K",
        "queen":  "Q", "rook":   "R",  "pawn":  "P",
    }

    white_pieces = {}
    black_pieces = {}
    for prefix, subdir, out in [("w", "white_pieces", white_pieces),
                                ("b", "black_pieces", black_pieces)]:
        d = os.path.join(assets_dir, subdir)
        for fname in os.listdir(d):
            if not fname.lower().endswith(".png"):
                continue
            kind = fname.split("_", 1)[1].rsplit(".", 1)[0]  # e.g. "bishop"
            code = piece_map.get(kind)
            if not code:
                continue
            surf = pygame.image.load(os.path.join(d, fname))
            surf = pygame.transform.smoothscale(surf, (scale_size, scale_size))
            out[f"{prefix}{code}"] = surf

    return white_pieces, black_pieces
