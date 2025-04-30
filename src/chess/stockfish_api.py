import requests
from typing import Optional, Tuple
import numpy as np


class StockfishAPIPlayer:
    API_URL = "https://stockfish.online/api/s/v2.php"

    def __init__(self, piece_state: np.ndarray, stockfish_depth: int):
        self.current_board = piece_state
        self.depth = stockfish_depth

    def get_stockfish_move(
            self,
            recent_piece_state: np.ndarray,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        fen = self._board_to_fen(recent_piece_state)

        try:
            resp = requests.get(
                self.API_URL,
                params={
                    "fen": fen,
                    "depth": self.depth
                },
                timeout=5
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Stockfish API request failed: {e}")
            return None

        if not data.get("success", False):
            print("Stockfish API returned error:", data)
            return None

        raw = data["bestmove"].split()
        if len(raw) < 2:
            print("Unexpected bestmove format:", data["bestmove"])
            return None
        uci_move = raw[1]

        return self._uci_to_coords(uci_move)

    def _board_to_fen(self, board: np.ndarray, to_move: str = 'b') -> str:
        ranks = []
        for row in board:
            empty = 0
            fen_row = ""
            for cell in row:
                if not cell:
                    empty += 1
                else:
                    if empty:
                        fen_row += str(empty)
                        empty = 0
                    color, kind = cell.split('-', 1)
                    letter = 'N' if kind == 'knight' else kind[0].upper()
                    fen_row += letter.lower() if color == 'black' else letter
            if empty:
                fen_row += str(empty)
            ranks.append(fen_row)
        placement = "/".join(ranks)
        return f"{placement} {to_move} KQkq - 0 1"

    def _uci_to_coords(self, uci: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        c_from = ord(uci[0]) - ord('a')
        r_from = 8 - int(uci[1])
        c_to = ord(uci[2]) - ord('a')
        r_to = 8 - int(uci[3])
        return (c_from, r_from), (c_to, r_to)
