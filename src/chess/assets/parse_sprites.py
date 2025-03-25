from PIL import Image


def parse_sprites() -> (dict, dict):
    """Parses the sprites from the spritesheet and returns a dictionary of sprites.

        white_pieces.png and black_pieces.png are ordered bishop (row 1), knight (row 2),
        king (row 3, col 1), queen (row 3, col 3), rook (row 3, col 4), pawn (row 4).
    """
    white_spritesheet = Image.open("white_pieces.png")
    black_spritesheet = Image.open("black_pieces.png")

    sprite_width, sprite_height = 32, 32

    white_sprites = {}
    black_sprites = {}

    cols = 128 // sprite_width
    rows = 128 // sprite_height

    # TODO: Refactor this below to properly parse the files (this does not work)
    for row in range(rows):
        for col in range(cols):
            left = col * sprite_width
            upper = row * sprite_height
            right = left + sprite_width
            lower = upper + sprite_height
            white_sprite = white_spritesheet.crop((left, upper, right, lower))
            black_sprite = black_spritesheet.crop((left, upper, right, lower))
            white_sprites[(row, col)] = white_sprite
            black_sprites[(row, col)] = black_sprite

    return white_sprites, black_sprites
