"""
Real-time chess piece detection and GUI visualization.

Opens two windows:
  1. OpenCV window – raw camera feed with bounding-box overlays.
  2. Pygame window   – 2-D board updated from detections.

Usage
-----

Press ‘r’ in the OpenCV window at any time to re-select the four board corners.
Press ‘q’ to quit.
"""
from __future__ import annotations

import collections
import threading
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pygame
from pygame import font
from ultralytics import YOLO

from .assets.parse_sprites import parse_sprites
from .stockfish_api import StockfishPlayer

_FILE_DIR = Path(__file__).resolve().parent

_PIECE_CODE_MAP = {
    "pawn": "P",
    "rook": "R",
    "knight": "Kn",
    "bishop": "B",
    "queen": "Q",
    "king": "K",
}

_PIECE_COLORS = {
    'bB': (128, 0, 128), 'bK': (0, 0, 180), 'bKn': (0, 128, 255),
    'bP': (0, 255, 255), 'bQ': (0, 0, 128), 'bR': (0, 255, 128),
    'wB': (0, 255, 0), 'wK': (255, 255, 255), 'wKn': (0, 165, 255),
    'wP': (0, 255, 255), 'wQ': (255, 0, 0), 'wR': (0, 215, 255)
}


def _fmt_ms(ms: int) -> str:
    mins, secs = divmod(ms // 1000, 60)
    return f"{mins}:{secs:02d}"


def _label_to_sprite_code(label: str) -> str | None:
    try:
        color, piece = label.split("-")
        return ("w" if color == "white" else "b") + _PIECE_CODE_MAP[piece]
    except Exception:
        return None


def _create_kalman_filter() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf


class RealtimeChessCV:
    """Encapsulates camera capture, YOLO inference, and GUI display."""

    # board & GUI constants
    _BOARD_PIX = 800
    _TILE_PIX = _BOARD_PIX // 8
    _TIMER_BAR = 80

    # OpenCV colours (BGR)
    _GRID_COLOUR = (0, 0, 255)

    def __init__(self, model_path: str | Path, camera_index: int = 0, stockfish_exe_path: str = "",
                 bounding_box_bottom_ratio: float = .90, white_mins: int = 5, black_mins: int = 5,
                 stockfish_elo: int = 2000):
        # Player logic
        self._turn = "white"
        self._white_ms = white_mins * 60_000
        self._black_ms = black_mins * 60_000
        self._timer_font = pygame.font.SysFont('Montserrat', 30)
        self._last_tick_ms = pygame.time.get_ticks()
        self._waiting_for_stockfish = False
        self._stockfish_arrow = None
        self._prev_np_board = np.full((8, 8), "", dtype=object)
        self._stockfish_player = StockfishPlayer(self._prev_np_board, stockfish_exe_path, stockfish_elo)
        self._bounding_box_bottom_ratio = bounding_box_bottom_ratio

        # Model and capture components
        self.model = YOLO(str(model_path))
        self.cap = cv2.VideoCapture(camera_index)

        # Synchronisation
        self._shared_piece_locations: List[dict] = []
        self._lock = threading.Lock()
        self._perspective_ready = threading.Event()

        # perspective matrices
        self._board_src: np.ndarray | None = None
        self._board_dst: np.ndarray | None = None
        self._M: np.ndarray | None = None

        # state caches
        self._history: Dict[tuple, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=10)
        )
        self._kalman_filters: Dict[tuple, cv2.KalmanFilter] = {}

        # pygame asseets
        self._white_sprites, self._black_sprites = parse_sprites(scale_size=self._TILE_PIX)
        self._sprite_dict = {**self._white_sprites, **self._black_sprites}

    def run(self) -> None:
        """Blocking main loop (runs until user presses ‘q’)."""
        # Step 1 – user clicks four board corners
        self._select_grid()

        # Step 2 – spin up Pygame GUI thread
        gui_thr = threading.Thread(target=self._pygame_loop, daemon=True)
        gui_thr.start()

        # Step 3 – main detection loop (OpenCV window)
        try:
            self._detection_loop()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            pygame.quit()

    def _tick_clock(self):
        now = pygame.time.get_ticks()
        delta = now - self._last_tick_ms
        self._last_tick_ms = now
        if self._turn == "white":
            with self._lock:
                self._white_ms = max(0, self._white_ms - delta)
        else:
            with self._lock:
                self._black_ms = max(0, self._black_ms - delta)

    def _select_grid(self) -> None:
        clicked_pts: List[List[int]] = []

        def _click_cb(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
                clicked_pts.append([x, y])
                print(f"✅ Point {len(clicked_pts)}: ({x}, {y})")

        cv2.namedWindow("Click Corners")
        cv2.setMouseCallback("Click Corners", _click_cb)
        print(
            "🖱 Click the 4 corners of the chessboard in this order:\n"
            "Top-left (A8), Top-right (H8), Bottom-left (A1), Bottom-right (H1)"
        )

        while len(clicked_pts) < 4:
            ok, frame = self.cap.read()
            if not ok:
                continue
            vis = frame.copy()
            for pt in clicked_pts:
                cv2.circle(vis, tuple(pt), 6, (0, 255, 255), -1)
            cv2.putText(
                vis,
                f"Click corner {len(clicked_pts) + 1}/4",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Click Corners", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt("Quit during board selection")

        cv2.destroyWindow("Click Corners")

        # compute perspective transform from canonical board to image space
        self._board_src = np.float32(
            [[0, 0], [self._BOARD_PIX, 0], [0, self._BOARD_PIX], [self._BOARD_PIX, self._BOARD_PIX]])
        self._board_dst = np.float32(clicked_pts)
        self._M = cv2.getPerspectiveTransform(
            self._board_dst,  # img→board (inverse of earlier code for GUI)
            self._board_src,
        )
        self._perspective_ready.set()

    def _detection_loop(self) -> None:
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break

            results = self.model.predict(source=frame, conf=0.7, iou=0.5, save=False, stream=True)

            annotated = frame.copy()
            current_pieces: List[dict] = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    label = self.model.names[cls]

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    center_id = (cx // 20, cy // 20)

                    # temporal majority vote for class stability
                    hist = self._history[center_id]
                    hist.append(cls)
                    most_common_cls, cnt = collections.Counter(hist).most_common(1)[0]
                    if cnt / len(hist) >= 0.7:
                        stable_cls = most_common_cls
                    else:
                        stable_cls = hist[-1]
                    label = self.model.names[stable_cls]

                    w, h = x2 - x1, y2 - y1

                    # Kalman smoothing per small region id
                    if center_id not in self._kalman_filters:
                        kf = _create_kalman_filter()
                        kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                        self._kalman_filters[center_id] = kf
                    corrected = self._kalman_filters[center_id].correct(
                        np.array([[np.float32(cx)], [np.float32(cy)]])
                    )
                    cx_s, cy_s = int(corrected[0][0]), int(corrected[1][0])

                    x1_s = cx_s - w // 2
                    y1_s = cy_s - h // 2

                    current_pieces.append(
                        {
                            "label": label,
                            "x1": x1_s,
                            "y1": y1_s,
                            "width": w,
                            "height": h,
                        }
                    )
                    label = label.replace("white", "w").replace("black", "b")
                    for k, v in _PIECE_CODE_MAP.items():
                        label = label.replace(k, v)
                    label = label.replace("-", "")

                    current_piece_color = _PIECE_COLORS[label]

                    # draw rectangle & label
                    cv2.rectangle(annotated, (x1_s, y1_s), (x1_s + w, y1_s + h), current_piece_color, 2)
                    cv2.putText(
                        annotated,
                        f"{label} {conf:.2f}",
                        (x1_s, y1_s - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        current_piece_color,
                        2,
                    )

            # push detections to GUI thread
            with self._lock:
                white_ms = self._white_ms
                black_ms = self._black_ms

                self._shared_piece_locations.clear()
                self._shared_piece_locations.extend(current_pieces)

            # overlay grid for user feedback
            self._draw_grid_on_frame(annotated)
            # Key shortcut info
            cv2.putText(
                annotated,
                "Press 'R' to redo grid | 'Space' for turn (GUI) | 'Q' to quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            # player times
            cv2.putText(
                annotated,
                f"White: {_fmt_ms(white_ms)} | Black: {_fmt_ms(black_ms)}",
                (10, self._BOARD_PIX + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Detection Feed", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self._perspective_ready.clear()
                self._select_grid()

    def _dict_to_np(self, state: Dict[str, str]) -> np.ndarray:
        """
        Convert an algebraic-notation dict (e.g. {'e4':'wP'}) → 8×8 ndarray.
        """
        board = np.full((8, 8), "", dtype=object)

        for sq, label in state.items():
            file_chr, rank_chr = sq[0], sq[1]
            col = "abcdefgh".index(file_chr)
            row = 8 - int(rank_chr)
            board[row, col] = label

        return board

    def _has_arrow_move_occurred(
            self,
            prev: np.ndarray,
            curr: np.ndarray,
    ) -> bool:
        """
        True ⇢ engine’s suggested piece has moved **exactly as drawn**.
        """
        if self._stockfish_arrow is None:
            return False

        (src_c, src_r), (dst_c, dst_r) = self._stockfish_arrow

        moved_piece = prev[src_r, src_c]
        if moved_piece == "":
            return False

        # basic source / destination sanity
        if curr[src_r, src_c] != "":
            return False
        if curr[dst_r, dst_c] != moved_piece:
            return False

        # optional strictness: make sure nothing else changed
        mask = np.ones(prev.shape, dtype=bool)
        mask[src_r, src_c] = False
        mask[dst_r, dst_c] = False
        if not np.array_equal(prev[mask], curr[mask]):
            return False

        return True

    def _handle_human_move(self):
        # freeze White’s clock
        self._tick_clock()
        self._turn = "black"

        with self._lock:
            detections = list(self._shared_piece_locations)

        state_dict = self._build_game_state(detections)
        self._prev_np_board = self._dict_to_np(state_dict)

        # ask Stockfish for a reply
        with self._lock:
            white_ms = self._white_ms
            black_ms = self._black_ms

        frm, to = self._stockfish_player.get_stockfish_move(
            self._prev_np_board, white_ms, black_ms
        )
        self._stockfish_arrow = (frm, to)
        self._waiting_for_stockfish = True

    def _pygame_loop(self) -> None:
        WIDTH = HEIGHT = self._BOARD_PIX
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess CV")

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and self._turn == "white"
                      and not self._waiting_for_stockfish):
                    self._handle_human_move()

            if not self._perspective_ready.is_set():
                pygame.display.flip()
                continue

            # copy detections under lock
            with self._lock:
                detections = list(self._shared_piece_locations)

            if self._waiting_for_stockfish:
                state_dict = self._build_game_state(detections)
                np_board = self._dict_to_np(state_dict)

                if self._has_arrow_move_occurred(self._prev_np_board, np_board):
                    # Black obeyed – stop their clock, start White’s
                    self._tick_clock()
                    self._turn = "white"
                    self._waiting_for_stockfish = False
                    self._stockfish_arrow = None
                    self._prev_np_board = np_board  # new baseline

            # render board & pieces
            screen.fill((0, 0, 0))
            self._draw_board(screen)
            state = self._build_game_state(detections)
            self._draw_pieces(screen, state)

            if self._stockfish_arrow:
                p1, p2 = self._stockfish_arrow
                s = self._TILE_PIX
                pygame.draw.line(
                    screen,
                    (255, 0, 0),
                    (p1[0] * s + s // 2, p1[1] * s + s // 2),
                    (p2[0] * s + s // 2, p2[1] * s + s // 2),
                    6
                )

            pygame.display.flip()
            clock.tick(10)

    def _draw_board(self, surface: pygame.Surface) -> None:
        light = (240, 217, 181)
        dark = (181, 136, 99)
        for row in range(8):
            for col in range(8):
                color = light if (row + col) % 2 == 0 else dark
                pygame.draw.rect(
                    surface,
                    color,
                    pygame.Rect(
                        col * self._TILE_PIX, row * self._TILE_PIX, self._TILE_PIX, self._TILE_PIX
                    ),
                )

    def _build_game_state(self, detections: List[dict]) -> Dict[str, str]:
        state: Dict[str, str] = {}
        if self._M is None:
            return state
        for piece in detections:
            cx = piece["x1"] + piece["width"] // 2
            cy = piece["y1"] + int(
                piece["height"] ** self._bounding_box_bottom_ratio)  # tweak for piece based on camera
            warped = cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float32), self._M
            )[0][0]
            gx, gy = warped
            col = int(gx // self._TILE_PIX)
            row = int(gy // self._TILE_PIX)
            if 0 <= col < 8 and 0 <= row < 8:
                square = f"{'abcdefgh'[col]}{8 - row}"
                state[square] = piece["label"]
        return state

    def _draw_pieces(self, surface: pygame.Surface, state: Dict[str, str]) -> None:
        for square, label in state.items():
            code = _label_to_sprite_code(label)
            sprite = self._sprite_dict.get(code)
            if sprite is None:
                continue  # silently skip unknowns
            col = "abcdefgh".index(square[0])
            row = 8 - int(square[1])
            x = col * self._TILE_PIX + (self._TILE_PIX - sprite.get_width()) // 2
            y = row * self._TILE_PIX + (self._TILE_PIX - sprite.get_height()) // 2
            surface.blit(sprite, (x, y))

    def _draw_grid_on_frame(self, frame: np.ndarray) -> None:
        if self._board_src is None or self._board_dst is None:
            return
        persp = cv2.getPerspectiveTransform(self._board_src, self._board_dst)
        for i in range(9):
            # vertical lines
            p1 = np.array([[[i * self._TILE_PIX, 0]]], dtype=np.float32)
            p2 = np.array([[[i * self._TILE_PIX, self._BOARD_PIX]]], dtype=np.float32)
            a = cv2.perspectiveTransform(p1, persp)[0][0]
            b = cv2.perspectiveTransform(p2, persp)[0][0]
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), self._GRID_COLOUR, 1)

            # horizontal lines
            p3 = np.array([[[0, i * self._TILE_PIX]]], dtype=np.float32)
            p4 = np.array([[[self._BOARD_PIX, i * self._TILE_PIX]]], dtype=np.float32)
            c = cv2.perspectiveTransform(p3, persp)[0][0]
            d = cv2.perspectiveTransform(p4, persp)[0][0]
            cv2.line(frame, tuple(c.astype(int)), tuple(d.astype(int)), self._GRID_COLOUR, 1)
