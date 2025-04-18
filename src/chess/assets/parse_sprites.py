import os
from PIL import Image
import cv2 as cv
import numpy as np


def parse_sprites(scale_size=32) -> tuple[dict, dict]:
    """Parses the sprites from the spritesheet and returns dictionaries of sprites.

    The spritesheets (white_pieces.png and black_pieces.png) contain:
    - Bishop: row 0 (4 variations)
    - Knight: row 1 (4 variations)
    - King (col 0), Queen (col 1), Rook (col 2): row 2
    - Pawn: row 3, col 0
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    white_pieces_path = os.path.join(current_dir, 'white_pieces.png')
    black_pieces_path = os.path.join(current_dir, 'black_pieces.png')

    white_spritesheet = cv.imread(white_pieces_path, cv.IMREAD_UNCHANGED)
    black_spritesheet = cv.imread(black_pieces_path, cv.IMREAD_UNCHANGED)

    if white_spritesheet.shape[2] == 3:
        white_spritesheet = cv.cvtColor(white_spritesheet, cv.COLOR_BGR2RGBA)
        black_spritesheet = cv.cvtColor(black_spritesheet, cv.COLOR_BGR2RGBA)

    sprite_width, sprite_height = 32, 32

    white_pieces = {}
    black_pieces = {}

    piece_positions = {
        'B': [(0, 0)],  # Bishop (first)
        'Kn': [(1, 0)],  # Knight (first)
        'K': [(2, 0)],  # King
        'Q': [(2, 1)],  # Queen
        'R': [(2, 2)],  # Rook (first)
        'P': [(3, 0)]  # Pawn
    }

    # Process each piece type
    for piece_type, positions in piece_positions.items():
        row, col = positions[0]

        y = row * sprite_height
        x = col * sprite_width

        white_sprite = white_spritesheet[y:y + sprite_height, x:x + sprite_width]
        black_sprite = black_spritesheet[y:y + sprite_height, x:x + sprite_width]

        white_mask = cv.inRange(white_sprite[:, :, :3], np.array([0, 0, 0]), np.array([10, 10, 10]))
        white_sprite[white_mask > 0] = [0, 0, 0, 0]  # Make black background transparent

        black_mask = cv.inRange(black_sprite[:, :, :3], np.array([128, 128, 0]), np.array([130, 130, 2]))
        black_sprite[black_mask > 0] = [0, 0, 0, 0]  # Make teal background transparent

        if scale_size != sprite_width:
            white_sprite = cv.resize(white_sprite, (scale_size, scale_size), interpolation=cv.INTER_AREA)
            black_sprite = cv.resize(black_sprite, (scale_size, scale_size), interpolation=cv.INTER_AREA)

        white_pieces[f"w{piece_type}"] = white_sprite
        black_pieces[f"b{piece_type}"] = black_sprite

    return white_pieces, black_pieces