from typing import Optional

import numpy as np

from stockfish import Stockfish, StockfishException


class StockfishPlayer:
    def __init__(self, piece_state: np.ndarray, stockfish_exe_path: str, stockfish_elo: int):
        self.current_board: np.ndarray = piece_state
        self.stockfish = Stockfish(
            path=stockfish_exe_path,
            parameters={
                "Threads": 2,
                "UCI_LimitStrength": False,
                "Skill Level": 20,
                "UCI_Elo": stockfish_elo
            })

    def get_stockfish_move(self, recent_piece_state: np.ndarray, white_time: int, black_time: int) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        best_move = None
        try:
            # Convert ChessBoardState to FEN
            fen = self._board_to_fen(recent_piece_state)
            print()
            # Update Stockfish with the current position
            self.stockfish.set_fen_position(fen)
            # Retrieve and return the best move
            best_move = self.stockfish.get_best_move(wtime=white_time, btime=black_time)
            print(f"original best_move: {best_move}")
        except StockfishException as e:
            print(f"Stockfish error: {e}")
        finally:
            resolved_best_move = self._to_and_from(best_move) if best_move else None
            print(f"resolved_best_move: {resolved_best_move}")
        return resolved_best_move

    def _board_to_fen(self, board: np.ndarray, to_move: str = 'b') -> str:
        rows = []
        for fen_rank in board:
            empty = 0
            row_s = ''
            for cell in fen_rank:
                if not cell:
                    empty += 1
                else:
                    if empty:
                        row_s += str(empty)
                        empty = 0
                    # split full name into color + piece
                    color, kind = cell.split('-', 1)
                    # map kind to letter
                    letter = 'N' if kind=='knight' else kind[0].upper()
                    # black pieces are lowercase
                    row_s += letter.lower() if color=='black' else letter
            if empty:
                row_s += str(empty)
            rows.append(row_s)
        placement = '/'.join(rows)

        return f"{placement} {to_move} KQkq - 0 1"

    def _to_and_from(self, best_move: str) -> tuple[tuple[int, int], tuple[int, int]]:
        if best_move is not None:
            col_from = ord(best_move[0]) - ord('a')
            row_from = 8 - int(best_move[1])
            col_to = ord(best_move[2]) - ord('a')
            row_to = 8 - int(best_move[3])
            return (col_from, row_from), (col_to, row_to)
