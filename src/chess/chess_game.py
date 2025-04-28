import math

import numpy as np
import pygame
from pygame import Surface, SurfaceType
from pygame.font import Font
from pygame.time import Clock

from stockfish_api import StockfishPlayer

from assets.parse_sprites import parse_sprites


class ChessPieces:
    def __init__(self):
        self.piece_names = ['K', 'Q', 'R', 'B', 'Kn', 'P']
        # Create an 8x8 array of empty strings.
        self.piece_state = np.full((8, 8), "", dtype=object)
        # Black pieces (top of board)
        self.piece_state[0] = np.array(["bR", "bKn", "bB", "bQ", "bK", "bB", "bKn", "bR"])
        self.piece_state[1] = np.array(["bP"] * 8)
        # White pieces (bottom of board)
        self.piece_state[6] = np.array(["wP"] * 8)
        self.piece_state[7] = np.array(["wR", "wKn", "wB", "wQ", "wK", "wB", "wKn", "wR"])

    def check_move_legality(self, piece: str, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a move from src to dst is legal for the given piece.
        """
        # First check: cannot move to a square with your own piece
        dst_piece = self.piece_state[dst[0], dst[1]]
        if dst_piece != "" and dst_piece[0] == piece[0]:
            return False

        if piece == "wP" or piece == "bP":
            return self._check_pawn_move(piece, src, dst)
        elif piece == "wR" or piece == "bR":
            return self._check_rook_move(src, dst)
        elif piece == "wKn" or piece == "bKn":
            return self._check_knight_move(src, dst)
        elif piece == "wB" or piece == "bB":
            return self._check_bishop_move(src, dst)
        elif piece == "wQ" or piece == "bQ":
            return self._check_queen_move(src, dst)
        elif piece == "wK" or piece == "bK":
            return self._check_king_move(src, dst)
        return False

    def _check_pawn_move(self, piece: str, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a pawn move from src to dst is legal.
        """
        # Determine direction based on pawn color.
        direction = -1 if piece[0] == 'w' else 1

        # Check for forward movement (can't capture)
        if dst[1] == src[1]:
            # Can only move forward to empty squares
            if self.piece_state[dst[0], dst[1]] != "":
                return False

            # Move one square forward
            if dst[0] == src[0] + direction:
                return True

            # Move two squares forward if the pawn is at the starting row
            if (src[0] == 1 and piece[0] == 'b') or (src[0] == 6 and piece[0] == 'w'):
                if dst[0] == src[0] + 2 * direction and self.piece_state[src[0] + direction, src[1]] == "":
                    return True
            return False

        # Check for diagonal capture
        if abs(dst[1] - src[1]) == 1 and dst[0] == src[0] + direction:
            # Must have an enemy piece to capture
            dst_piece = self.piece_state[dst[0], dst[1]]
            if dst_piece != "" and dst_piece[0] != piece[0]:
                return True

        # En passant would be handled here
        return False

    def _check_rook_move(self, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a rook move from src to dst is legal.
        """
        # Rook moves horizontally or vertically.
        if src[0] == dst[0] or src[1] == dst[1]:
            # Check if there are any pieces in the way.
            if src[0] == dst[0]:
                # Move is horizontal.
                start = min(src[1], dst[1])
                end = max(src[1], dst[1])
                for j in range(start + 1, end):
                    if self.piece_state[src[0], j] != "":
                        return False
            else:
                # Move is vertical.
                start = min(src[0], dst[0])
                end = max(src[0], dst[0])
                for i in range(start + 1, end):
                    if self.piece_state[i, src[1]] != "":
                        return False
            return True
        return False

    def _check_knight_move(self, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a knight move from src to dst is legal.
        """
        # Knight moves in an L-shape: two squares in one direction and one square in the other.
        if (abs(dst[0] - src[0]) == 2 and abs(dst[1] - src[1]) == 1) or (
                abs(dst[0] - src[0]) == 1 and abs(dst[1] - src[1]) == 2):
            return True
        return False

    def _check_bishop_move(self, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a bishop move from src to dst is legal.
        """
        # Bishop moves diagonally.
        if abs(dst[0] - src[0]) == abs(dst[1] - src[1]):
            # Determine the direction of movement
            row_step = 1 if dst[0] > src[0] else -1
            col_step = 1 if dst[1] > src[1] else -1

            # Check if there are any pieces in the way
            row, col = src[0] + row_step, src[1] + col_step
            while (row, col) != dst:
                if self.piece_state[row, col] != "":
                    return False
                row += row_step
                col += col_step
            return True
        return False

    def _check_queen_move(self, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a queen move from src to dst is legal.
        """
        # Queen combines rook and bishop movement
        # For simplicity, we reuse the rook and bishop logic
        if src[0] == dst[0] or src[1] == dst[1]:
            return self._check_rook_move(src, dst)
        elif abs(dst[0] - src[0]) == abs(dst[1] - src[1]):
            return self._check_bishop_move(src, dst)
        return False

    def _check_king_move(self, src: tuple[int, int], dst: tuple[int, int]):
        """
        Check if a king move from src to dst is legal.
        """
        # King moves one square in any direction.
        if abs(dst[0] - src[0]) <= 1 and abs(dst[1] - src[1]) <= 1:
            return True
        return False


class ChessBoard:
    def __init__(self):
        # Create an 8x8 board for tile colors.
        self.board_tiles = np.zeros((8, 8), dtype=int)
        # Alternate tile colors: we use 1 for light and 0 for dark.
        for i in range(8):
            for j in range(8):
                self.board_tiles[i, j] = (i + j) % 2


class ChessBoardState:
    def __init__(self):
        self.board = ChessBoard()
        self.pieces = ChessPieces()
        # Track whose turn it is: 'w' for white, 'b' for black.
        self.current_turn = 'w'
        # For en passant: store a tuple (piece, src, dst) of the last move.
        self.last_move = None

    def update_state(self, src: tuple[int, int], dst: tuple[int, int]):
        """
        Move a piece from src (row, col) to dst (row, col).
        Only moves a piece if it belongs to the current player.
        (No full move legality is checked here.)
        """
        piece = self.pieces.piece_state[src[0], src[1]]
        if piece == "" or piece[0] != self.current_turn:
            return False

        # Move piece (capture any piece on destination)
        self.pieces.piece_state[dst[0], dst[1]] = piece
        self.pieces.piece_state[src[0], src[1]] = ""
        self.last_move = (piece, src, dst)
        # Switch turn
        self.current_turn = 'b' if self.current_turn == 'w' else 'w'
        return True

    def promote_piece(self, pos: tuple[int, int], new_type: str = "Q"):
        """
        Promote a pawn at pos (row, col) to a new piece type.
        By default, promotion is to a Queen.
        """
        piece = self.pieces.piece_state[pos[0], pos[1]]
        if piece != "" and piece[1] == "P":
            self.pieces.piece_state[pos[0], pos[1]] = piece[0] + new_type

    def castle_kingside(self):
        """
        If castling is available for the current turn,
        move the king two squares and the rook to the square adjacent.
        (This simplified version only checks for the standard starting positions
         and that the intervening squares are empty.)
        """
        if self.current_turn == 'w':
            # White king must be at (7,4) and rook at (7,7)
            if self.pieces.piece_state[7, 4] == "wK" and self.pieces.piece_state[7, 7] == "wR":
                if self.pieces.piece_state[7, 5] == "" and self.pieces.piece_state[7, 6] == "":
                    # Perform castling: king goes to (7,6) and rook to (7,5)
                    self.pieces.piece_state[7, 6] = "wK"
                    self.pieces.piece_state[7, 4] = ""
                    self.pieces.piece_state[7, 5] = "wR"
                    self.pieces.piece_state[7, 7] = ""
                    self.current_turn = 'b'
                    return True
        else:
            # Black king must be at (0,4) and rook at (0,7)
            if self.pieces.piece_state[0, 4] == "bK" and self.pieces.piece_state[0, 7] == "bR":
                if self.pieces.piece_state[0, 5] == "" and self.pieces.piece_state[0, 6] == "":
                    self.pieces.piece_state[0, 6] = "bK"
                    self.pieces.piece_state[0, 4] = ""
                    self.pieces.piece_state[0, 5] = "bR"
                    self.pieces.piece_state[0, 7] = ""
                    self.current_turn = 'w'
                    return True
        return False

    def castle_queenside(self):
        """
        Perform queenside castling if available.
        King moves two squares toward the rook; the rook jumps to the square adjacent.
        """
        if self.current_turn == 'w':
            # White king must be at (7,4) and rook at (7,0)
            if self.pieces.piece_state[7, 4] == "wK" and self.pieces.piece_state[7, 0] == "wR":
                if self.pieces.piece_state[7, 1] == "" and self.pieces.piece_state[7, 2] == "" and \
                        self.pieces.piece_state[7, 3] == "":
                    self.pieces.piece_state[7, 2] = "wK"
                    self.pieces.piece_state[7, 4] = ""
                    self.pieces.piece_state[7, 3] = "wR"
                    self.pieces.piece_state[7, 0] = ""
                    self.current_turn = 'b'
                    return True
        else:
            # Black king must be at (0,4) and rook at (0,0)
            if self.pieces.piece_state[0, 4] == "bK" and self.pieces.piece_state[0, 0] == "bR":
                if self.pieces.piece_state[0, 1] == "" and self.pieces.piece_state[0, 2] == "" and \
                        self.pieces.piece_state[0, 3] == "":
                    self.pieces.piece_state[0, 2] = "bK"
                    self.pieces.piece_state[0, 4] = ""
                    self.pieces.piece_state[0, 3] = "bR"
                    self.pieces.piece_state[0, 0] = ""
                    self.current_turn = 'w'
                    return True
        return False

    def en_passant(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        """
        Perform en passant capture if applicable.
        src: starting position (row, col) of the pawn making the capture.
        dst: destination position (row, col) where the pawn lands.
        """
        piece = self.pieces.piece_state[src[0], src[1]]
        if piece == "" or piece[1] != "P":
            return False

        direction = -1 if piece[0] == 'w' else 1
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]

        if dr != direction or abs(dc) != 1:
            return False

        if self.pieces.piece_state[dst[0], dst[1]] != "":
            return False

        if not self.last_move:
            return False
        last_piece, last_src, last_dst = self.last_move

        if last_piece[1] != "P" or last_piece[0] == piece[0]:
            return False

        if abs(last_dst[0] - last_src[0]) != 2:
            return False

        if last_dst[0] != src[0] or last_dst[1] != dst[1]:
            return False

        self.pieces.piece_state[dst[0], dst[1]] = piece
        self.pieces.piece_state[src[0], src[1]] = ""
        self.pieces.piece_state[src[0], dst[1]] = ""
        self.last_move = (piece, src, dst)

        self.current_turn = 'b' if self.current_turn == 'w' else 'w'
        return True

    def _is_in_check(self, color: str) -> bool:
        """Return True if the king of `color` is under attack."""
        king_pos = None
        for i in range(8):
            for j in range(8):
                if self.pieces.piece_state[i, j] == f"{color}K":
                    king_pos = (i, j)
                    break
            if king_pos:
                break

        if not king_pos:
            # no king on board? treat as in check
            return True

        enemy = 'b' if color == 'w' else 'w'
        for i in range(8):
            for j in range(8):
                p = str(self.pieces.piece_state[i, j])
                if p.startswith(enemy):
                    if self.pieces.check_move_legality(p, (i, j), king_pos):
                        return True
        return False

    def _has_any_safe_move(self, color: str) -> bool:
        """
        Return True if `color` has at least one legal move that
        does NOT leave their king in check.
        """
        orig_state = self.pieces.piece_state
        orig_array = orig_state.copy()

        # locate king once
        king_pos = None
        for i in range(8):
            for j in range(8):
                if orig_array[i, j] == f"{color}K":
                    king_pos = (i, j)
                    break
            if king_pos:
                break

        enemy = 'b' if color == 'w' else 'w'

        # try every piece and every destination
        for r0 in range(8):
            for c0 in range(8):
                p = orig_array[r0, c0]
                if not p.startswith(color):
                    continue

                for r1 in range(8):
                    for c1 in range(8):
                        if (r0, c0) == (r1, c1):
                            continue
                        if not self.pieces.check_move_legality(p, (r0, c0), (r1, c1)):
                            continue

                        new_array = orig_array.copy()
                        new_array[r1, c1] = p
                        new_array[r0, c0] = ""
                        self.pieces.piece_state = new_array

                        test_king = (r1, c1) if p == f"{color}K" else king_pos

                        still_in_check = self._is_in_check(color)
                        # restore
                        self.pieces.piece_state = orig_array
                        if not still_in_check:
                            return True
        return False

    def is_in_check(self, color: str) -> bool:
        if self._is_in_check(color):
            return True
        return False

    def check_for_checkmate(self) -> bool:
        """True if current player is in check and has no legal escape."""
        color = self.current_turn
        if not self._is_in_check(color):
            return False
        return not self._has_any_safe_move(color)

    def check_for_stalemate(self) -> bool:
        """
        True if current player is *not* in check but has no legal moves at all.
        """
        color = self.current_turn
        if self._is_in_check(color):
            return False
        return not self._has_any_safe_move(color)


class TimerInputScreen:
    def __init__(self):
        self.minutes = 10  # Default of 10 minutes; is editable from GUI
        self.font = pygame.font.SysFont(None, 40)
        self.title_font = pygame.font.SysFont(None, 90)
        self.input_active = False
        self.input_text = str(self.minutes)
        self.cursor_visible = True
        self.cursor_timer = 0

    def run(self, screen):
        """Run the timer input screen and return the selected time in seconds"""
        clock = pygame.time.Clock()
        running = True
        screen_width, screen_height = screen.get_size()
        center_x = screen_width // 2

        while running:
            # cursor blinking
            self.cursor_timer += clock.get_time()
            if self.cursor_timer >= 500:  # Toggle cursor every 500ms
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    input_rect = pygame.Rect(center_x - 100, 400, 200, 50)

                    # Check if input box is clicked
                    if input_rect.collidepoint(event.pos):
                        if not self.input_active:
                            self.input_active = True
                            self.input_text = ""
                    else:
                        self.input_active = False

                    # Check if start button is clicked
                    start_button = pygame.Rect(center_x - 100, 500, 200, 50)
                    if start_button.collidepoint(event.pos):
                        try:
                            minutes = int(self.input_text) if self.input_text else 10
                            self.minutes = max(1, minutes)
                            running = False
                        except ValueError:
                            self.input_text = "10"

                elif event.type == pygame.KEYDOWN and self.input_active:
                    if event.key == pygame.K_RETURN:
                        try:
                            minutes = int(self.input_text) if self.input_text else 10
                            self.minutes = max(1, minutes)
                            running = False
                        except ValueError:
                            self.input_text = "10"
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    elif event.unicode.isdigit():
                        self.input_text += event.unicode

            # Draw background
            screen.fill((77, 77, 77))

            # Draw title
            title = self.title_font.render("Chess", True, (255, 255, 255))
            title_rect = title.get_rect(center=(center_x, 200))
            screen.blit(title, title_rect)

            # Draw input label
            label = self.font.render("Minutes per player:", True, (255, 255, 255))
            label_rect = label.get_rect(center=(center_x, 350))
            screen.blit(label, label_rect)

            # Draw input box
            input_rect = pygame.Rect(center_x - 100, 400, 200, 50)
            pygame.draw.rect(screen, (200, 200, 200), input_rect)
            pygame.draw.rect(screen, (0, 0, 0), input_rect, 2)

            # Draw input text
            display_text = self.input_text
            if self.input_active and self.cursor_visible:
                display_text += "|"  # blink

            text_surface = self.font.render(display_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=input_rect.center)
            screen.blit(text_surface, text_rect)

            # Draw start button
            start_button = pygame.Rect(center_x - 100, 500, 200, 50)
            pygame.draw.rect(screen, (0, 200, 0), start_button)
            start_text = self.font.render("Start Game", True, (255, 255, 255))
            start_text_rect = start_text.get_rect(center=start_button.center)
            screen.blit(start_text, start_text_rect)

            pygame.display.flip()
            clock.tick(30)

        return self.minutes * 60


def format_time(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def np_to_surface(img_array: np.ndarray) -> pygame.Surface:
    height, width = img_array.shape[:2]
    surface = pygame.image.frombuffer(img_array.tobytes(), (width, height), 'RGBA')
    return surface.convert_alpha()


class ChessGame:
    def __init__(self):
        self.board_state = ChessBoardState()
        self.piece_selection: tuple[int, int] = None  # For selecting a piece to move (source square).
        self.square_size = 100
        self.white_pieces, self.black_pieces = parse_sprites()

        self.white_captured = []
        self.black_captured = []

        self.CAPTURED_PIECE_SIZE = 30
        self.MAX_PIECES_PER_ROW = 8

        self.white_time = 0
        self.black_time = 0
        self.last_time = 0

        self.game_over = False
        self.winner = None

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 40)
        self.timer_font = pygame.font.SysFont(None, 36)

    def move_piece(self, src: tuple[int,int], dst: tuple[int,int]):
        piece = self.board_state.pieces.piece_state[src[0], src[1]]
        # 1) Castling
        if piece and piece[1] == 'K' and abs(dst[1] - src[1]) == 2:
            # kingside
            if dst[1] - src[1] == 2 and self.board_state.castle_kingside():
                return
            # queenside
            if dst[1] - src[1] == -2 and self.board_state.castle_queenside():
                return
        # 2) En passant
        if piece and piece[1] == 'P':
            captured = str(self.board_state.pieces.piece_state[src[0], dst[1]])
            if self.board_state.en_passant(src, dst):
                if captured:
                    if captured.startswith('w'):
                        self.black_captured.append(captured)
                    else:
                        self.white_captured.append(captured)
                return
        # 3) Normal capture
        dst_piece = str(self.board_state.pieces.piece_state[dst[0], dst[1]])
        if dst_piece:
            if dst_piece.startswith('w'):
                self.black_captured.append(dst_piece)
            else:
                self.white_captured.append(dst_piece)
        # 4) Update state and record
        self.board_state.update_state(src, dst)

    def draw_captured_pieces(self, screen):
        """Draw captured pieces in the timer sections."""
        screen_width = screen.get_width()
        # white captures for black player
        self._draw_player_captures(screen, self.white_captured, screen_width - 260, 20)
        # black captures for white player
        screen_height = screen.get_height()
        self._draw_player_captures(screen, self.black_captured, screen_width - 260, screen_height - 80)

    def _draw_player_captures(self, screen, captured_pieces, start_x, start_y):
        """Helper to draw one player's captured pieces."""
        for i, piece in enumerate(captured_pieces):
            row = i // self.MAX_PIECES_PER_ROW
            col = i % self.MAX_PIECES_PER_ROW

            x = start_x + col * self.CAPTURED_PIECE_SIZE
            y = start_y + row * self.CAPTURED_PIECE_SIZE

            try:
                if piece.startswith('w'):
                    piece_img = self.white_pieces[piece]
                else:
                    piece_img = self.black_pieces[piece]

                scaled_img = pygame.transform.scale(piece_img,
                                                    (self.CAPTURED_PIECE_SIZE, self.CAPTURED_PIECE_SIZE))

                # Draw the piece
                screen.blit(scaled_img, (x, y))

            except KeyError:
                text = self.font.render(piece[1:], True, (255, 255, 255))
                text_rect = text.get_rect(center=(x + self.CAPTURED_PIECE_SIZE // 2,
                                                  y + self.CAPTURED_PIECE_SIZE // 2))
                screen.blit(text, text_rect)

    def _draw_stockfish_move_arrow(self, screen, source_x, source_y, target_x, target_y):
        head_len = 20
        color = (200, 30, 30)
        width = 5

        # main shaft
        pygame.draw.line(screen, color, (source_x, source_y), (target_x, target_y), width)

        # arrowhead
        angle = math.atan2(target_y - source_y, target_x - source_x)
        left = (target_x - head_len * math.cos(angle - math.pi/6),
                 target_y - head_len * math.sin(angle - math.pi/6))
        right = (target_x - head_len * math.cos(angle + math.pi/6),
                 target_y - head_len * math.sin(angle + math.pi/6))
        pygame.draw.polygon(screen, color, [(target_x, target_y), left, right])

    def run_game(self):
        pygame.init()

        # Create a screen with extra space for timers (100px at top and bottom)
        screen_height = 800 + 100 + 100  # Board height + top timer + bottom timer
        screen = pygame.display.set_mode((800, screen_height))
        pygame.display.set_caption('Chess Game GUI')

        # Convert piece icon arrays to pygame surface
        self.white_pieces = {k: np_to_surface(v) for k, v in self.white_pieces.items()}
        self.black_pieces = {k: np_to_surface(v) for k, v in self.black_pieces.items()}

        # Get timer settings from input screen
        timer_input = TimerInputScreen()
        initial_time = timer_input.run(screen)
        if initial_time is None:  # User closed the window
            return

        self.white_time = initial_time
        self.black_time = initial_time

        self.white_castled = False
        self.black_castled = False

        clock: Clock = pygame.time.Clock()
        running = True
        self.last_time = pygame.time.get_ticks()

        self.check_status = False

        self.black_stockfish_player = StockfishPlayer(self.board_state.pieces.piece_state)

        while running and not self.game_over:
            current_time = pygame.time.get_ticks()
            time_delta = (current_time - self.last_time) / 1000  # Converts to seconds
            self.last_time = current_time

            # Update the active player's timer
            if self.board_state.current_turn == 'w':
                self.white_time -= time_delta
                if self.white_time <= 0:
                    self.white_time = 0
                    self.game_over = True
            else:
                self.black_time -= time_delta
                if self.black_time <= 0:
                    self.black_time = 0
                    self.game_over = True

            for event in pygame.event.get():
                if self.board_state.current_turn == 'w':
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        adjusted_y = y - 100

                        # only process clicks within the board area
                        if 0 <= x < 800 and 0 <= adjusted_y < 800:
                            col = x // self.square_size
                            row = adjusted_y // self.square_size
                            if self.piece_selection is None:
                                piece = str(self.board_state.pieces.piece_state[row, col])

                                if piece and piece.startswith(self.board_state.current_turn):
                                    self.piece_selection = (row, col)
                            else:
                                src = self.piece_selection
                                dst = (row, col)
                                piece = str(self.board_state.pieces.piece_state[src])
                                moved = False

                                # 1) Castling
                                current_turn = self.board_state.current_turn
                                if not self.white_castled and current_turn == 'w':
                                    if piece[1] == 'K' and abs(dst[1] - src[1]) == 2:
                                        if dst[1] > src[1]:
                                            self.castle_kingside()
                                            moved = True
                                        else:
                                            self.castle_queenside()
                                            moved = True
                                    self.white_castled = True

                                elif not self.black_castled and current_turn == 'b':
                                    if piece[1] == 'K' and abs(dst[1] - src[1]) == 2:
                                        if dst[1] > src[1]:
                                            self.castle_kingside()
                                            moved = True
                                        else:
                                            self.castle_queenside()
                                            moved = True
                                    self.black_castled = True

                                # 2) En passant
                                if not moved and piece[1] == 'P' and \
                                        self.board_state.pieces.piece_state[dst] == "" and \
                                        dst[0] == src[0] + (-1 if piece[0] == 'w' else 1) and \
                                        abs(dst[1] - src[1]) == 1:
                                    moved = self.board_state.en_passant(src, dst)

                                # 3) Normal move (with check status checking)
                                if not moved and self.check_move_legality(piece, src, dst):
                                    # snapshot everything we’ll need to restore
                                    old_board = self.board_state.pieces.piece_state.copy()
                                    old_last_move = self.board_state.last_move
                                    old_turn = current_turn

                                    # apply the move (this flips current_turn internally)
                                    self.move_piece(src, dst)

                                    # check whether *that same color* is in check
                                    if self.check('w' if self.board_state.current_turn == 'b' else 'b'):
                                        self.board_state.pieces.piece_state = old_board
                                        self.board_state.last_move = old_last_move
                                        self.board_state.current_turn = old_turn
                                    else:
                                        moved = True
                                        self.check_status = False

                                self.piece_selection = None

                                # after ANY legal move, test for opponent check or end-game
                                if moved:
                                    if self.check('w' if self.board_state.current_turn == 'w' else 'b'):
                                        self.check_status = True

                                    if self.checkmate():
                                        self.game_over = True
                                        self.winner = "Black" if self.board_state.current_turn == 'w' else "White"

                                    elif self.stalemate():
                                        self.game_over = True
                                        self.winner = "Draw"
                elif self.board_state.current_turn == 'b':
                    source_and_dest = self.black_stockfish_player.get_stockfish_move(self.board_state.pieces.piece_state, self.white_time, self.black_time)
                    source_x, source_y = source_and_dest[0][0], source_and_dest[0][1]
                    target_x, target_y = source_and_dest[1][0], source_and_dest[1][1]

                    self._draw_stockfish_move_arrow(screen, source_x, source_y, target_x, target_y)
                    self.move_piece((source_x, source_y), (target_x, target_y))

            # Clear the screen
            screen.fill((30, 30, 30))

            pygame.draw.rect(screen, (77, 77, 77), pygame.Rect(0, 0, 800, 100))
            black_timer = self.timer_font.render(f"Black: {format_time(int(self.black_time))}", True,
                                                 (255, 255, 255))
            screen.blit(black_timer, (350, 40))

            pygame.draw.rect(screen, (77, 77, 77), pygame.Rect(0, 900, 800, 100))
            white_timer = self.timer_font.render(f"White: {format_time(int(self.white_time))}", True,
                                                 (255, 255, 255))
            screen.blit(white_timer, (350, 940))

            turn_text = f"Current turn: {'Check!' if self.check_status else ''}"
            turn_indicator = self.timer_font.render(turn_text, True, (255, 255, 0))
            screen.blit(turn_indicator, (50, 40 if self.board_state.current_turn == 'b' else 940))

            # Draw the board
            for i in range(8):
                for j in range(8):
                    # Use a light color for even tiles and a dark color for odd tiles.
                    tile_color = (238, 238, 210) if (i + j) % 2 == 0 else (118, 150, 86)
                    rect = pygame.Rect(j * self.square_size, i * self.square_size + 100,  # Offset by top timer height
                                       self.square_size, self.square_size)
                    pygame.draw.rect(screen, tile_color, rect)

                    # Draw a selection highlight if this square is selected.
                    if self.piece_selection == (i, j):
                        pygame.draw.rect(screen, (200, 0, 0), rect, 3)

                    # Draw any piece that exists in this square.
                    piece: str = str(self.board_state.pieces.piece_state[i, j])
                    if piece != "":
                        if piece.startswith("w"):
                            piece_img = self.white_pieces[piece]
                        else:
                            piece_img = self.black_pieces[piece]

                        scaled_img = pygame.transform.smoothscale(piece_img, (self.square_size, self.square_size))
                        img_rect = scaled_img.get_rect(center=rect.center)
                        screen.blit(scaled_img, img_rect)

            self.draw_captured_pieces(screen)

            if self.game_over:
                game_over_surface = pygame.Surface((400, 200), pygame.SRCALPHA)
                game_over_surface.fill((0, 0, 0, 200))
                game_over_text = self.timer_font.render(f"Game Over! {self.winner} wins!" if self.winner != "Draw"
                                                        else f"Game Over! {self.winner}", True, (255, 255, 255))
                game_over_surface.blit(game_over_text, (50, 80))
                screen.blit(game_over_surface, (200, 350))

            pygame.display.flip()
            clock.tick(30)

        if self.game_over:
            pygame.time.wait(3000)

        pygame.quit()

    def get_board_state(self):
        return self.board_state

    def check_move_legality(self, piece, row, col) -> bool:
        return self.board_state.pieces.check_move_legality(piece, row, col)

    def promote_piece(self, pos: tuple[int, int], new_type: str = "Q"):
        self.board_state.promote_piece(pos, new_type)

    def castle_kingside(self):
        self.board_state.castle_kingside()

    def castle_queenside(self):
        self.board_state.castle_queenside()

    def en_passant(self, src: tuple[int, int], dst: tuple[int, int]):
        self.board_state.en_passant(src, dst)

    def check(self, color: str):
        return self.board_state.is_in_check(color)

    def checkmate(self):
        return self.board_state.check_for_checkmate()

    def stalemate(self):
        return self.board_state.check_for_stalemate()


if __name__ == '__main__':
    game = ChessGame()
    game.run_game()
