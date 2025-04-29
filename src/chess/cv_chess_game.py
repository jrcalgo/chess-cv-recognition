import math
from typing import Optional

import numpy as np
import pygame
from pygame.time import Clock
import cv2

from src.chess.ui.chess_game_panel import ChessBoardState, TimerInputScreen
from src.chess.ui.computer_vision_panel import ComputerVisionPanel, AnnotationAggregator, bgr2rgb
from .stockfish_api import StockfishPlayer
from .assets.parse_sprites import parse_sprites


def format_time(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def np_to_surface(img_array: np.ndarray) -> pygame.Surface:
    height, width = img_array.shape[:2]

    if img_array.ndim == 3 and img_array.shape[2] == 3:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)
        surface = pygame.image.frombuffer(rgb.tobytes(), (width, height), 'RGBA')
        return surface.convert()
    elif img_array.ndim == 3 and img_array.shape[2] == 4:
        surface = pygame.image.frombuffer(img_array.tobytes(), (width, height), 'RGBA')
        return surface.convert_alpha()


def fit_to_scale(surface, target_rect):
    target_w, target_h = target_rect.size
    origin_w, origin_h = surface.get_width(), surface.get_height()
    scale = min(target_w / origin_w, target_h / origin_h)
    new_w, new_h = int(scale * origin_w), int(scale * origin_h)
    scaled = pygame.transform.smoothscale(surface, (new_w, new_h))
    x = target_rect.x + (target_w - new_w) // 2
    y = target_rect.y + (target_h - new_h) // 2
    return scaled, x, y


class CVChessGame:
    def __init__(self, model_path: str, video_capture_device: int, capture_orientation: str = 'landscape'):
        self.board_state = ChessBoardState()
        self.piece_selection: Optional[tuple[int, int]] = None
        self.square_size = 100
        self.white_pieces, self.black_pieces = parse_sprites(scale_size=self.square_size)

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
        self.font = pygame.font.SysFont('Montserrat', 40)
        self.timer_font = pygame.font.SysFont('Montserrat', 36)
        self.box_text_font = pygame.font.SysFont('Montserrat', 25)

        # Initialize camera capture panel
        display_size = (800, 1000)
        self.cv_panel = ComputerVisionPanel(display_size, model_path, video_capture_device, capture_orientation)
        self.last_annotated = None
        self.annotation_aggregator = AnnotationAggregator(n_annotations=50)

    def move_piece(self, src: tuple[int, int], dst: tuple[int, int]):
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
        left = (target_x - head_len * math.cos(angle - math.pi / 6),
                target_y - head_len * math.sin(angle - math.pi / 6))
        right = (target_x - head_len * math.cos(angle + math.pi / 6),
                 target_y - head_len * math.sin(angle + math.pi / 6))
        pygame.draw.polygon(screen, color, [(target_x, target_y), left, right])

    def run_game(self):
        screen_height = 800 + 100 + 100  # Board height + top timer + bottom timer
        width = 1600
        screen = pygame.display.set_mode((width, screen_height))
        pygame.display.set_caption('Chess CV')
        half_width = width // 2

        # Get timer settings from input screen
        timer_input = TimerInputScreen()
        initial_time = timer_input.run(screen)
        if initial_time is None:  # User closed the window
            return

        old_annotation_surface = None

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
                    def _square_location(row, col):
                        x = col * self.square_size + self.square_size // 2
                        y = row * self.square_size + self.square_size // 2
                        return x, y

                    source_and_dest = self.black_stockfish_player.get_stockfish_move(
                        self.board_state.pieces.piece_state, self.white_time, self.black_time)
                    source_x, source_y = _square_location(source_and_dest[0][0], source_and_dest[0][1])
                    target_x, target_y = _square_location(source_and_dest[1][0], source_and_dest[1][1])

                    self._draw_stockfish_move_arrow(screen, source_x, source_y, target_x, target_y)

            # Clear the screen
            screen.fill((30, 30, 30))

            # Render ComputerVisionPanel frames from cv and model
            frame, rect, text = self.cv_panel.pull_for_render()
            if frame is not None:
                annotation_frame = frame.copy()

                annotation_frame = cv2.cvtColor(annotation_frame, cv2.COLOR_BGR2RGB)
                annotation_frame = cv2.resize(annotation_frame, (800, 1000), interpolation=cv2.INTER_LINEAR)

                annotated_surface = pygame.image.frombuffer(annotation_frame.tobytes(), (800, 1000), 'RGB').convert()

                if rect is not None:
                    inf_h, inf_w = frame.shape[:2]
                    x_scale = half_width / inf_w
                    y_scale = screen_height / inf_h

                    scaled_rects = [((x1*x_scale, y1*y_scale), (x2*x_scale, y2*y_scale), color, thickness)
                                    for (x1, y1), (x2, y2), color, thickness in rect]

                    scaled_texts = [(text, (x*x_scale, y*y_scale), color) for text, (x, y), color in text]

                    for (x1, y1), (x2, y2), color, thickness in scaled_rects:
                        color = bgr2rgb(color)
                        w, h = x2 - x1, y2 - y1
                        pygame.draw.rect(annotated_surface, color, (x1, y1, w, h), thickness)

                    for text, (x, y), color in scaled_texts:
                        color = bgr2rgb(color)
                        text_surface = self.box_text_font.render(text, True, color)
                        text_surface.set_colorkey((0, 0, 0))
                        annotated_surface.blit(text_surface, (x, y))

                old_annotation_surface = annotated_surface.copy()
                scaled, x, y = fit_to_scale(annotated_surface, pygame.Rect(0, 0, half_width, screen_height))
                screen.blit(scaled, (x, y))
            elif old_annotation_surface is not None:
                scaled, x, y = fit_to_scale(old_annotation_surface, pygame.Rect(0, 0, half_width, screen_height))
                screen.blit(scaled, (x, y))

            # Draw timers and turn indicators, shifted right by half_width
            pygame.draw.rect(screen, (77, 77, 77), pygame.Rect(half_width, 0, 800, 100))
            black_timer = self.timer_font.render(f"Black: {format_time(int(self.black_time))}", True,
                                                 (255, 255, 255))
            screen.blit(black_timer, (half_width + 350, 40))

            pygame.draw.rect(screen, (77, 77, 77), pygame.Rect(half_width, 900, 800, 100))
            white_timer = self.timer_font.render(f"White: {format_time(int(self.white_time))}", True,
                                                 (255, 255, 255))
            screen.blit(white_timer, (half_width + 350, 940))

            turn_text = f"Current turn: {'Check!' if self.check_status else ''}"
            turn_indicator = self.timer_font.render(turn_text, True, (255, 255, 0))
            screen.blit(turn_indicator, (half_width + 50, 40 if self.board_state.current_turn == 'b' else 940))

            # Draw the board, shifted right by half_width
            for i in range(8):
                for j in range(8):
                    # Use a light color for even tiles and a dark color for odd tiles.
                    tile_color = (238, 238, 210) if (i + j) % 2 == 0 else (118, 150, 86)
                    rect = pygame.Rect(j * self.square_size + half_width, i * self.square_size + 100,
                                       # Offset by top timer height
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
                screen.blit(game_over_surface, (half_width + 200, 350))

            pygame.display.flip()
            clock.tick(60)

        if self.game_over:
            pygame.time.wait(10000)

        self.cv_panel.quit()
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
