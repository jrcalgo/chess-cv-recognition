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

    def get_stockfish_move(self, recent_piece_state: np.ndarray, white_time: int, black_time: int) -> tuple[tuple[int, int], tuple[int, int]]:
        best_move = None
        try:
            # Convert ChessBoardState to FEN
            fen = self._board_to_fen(recent_piece_state)
            # Update Stockfish with the current position
            self.stockfish.set_fen_position(fen)
            # Retrieve and return the best move
            best_move = self.stockfish.get_best_move(wtime=white_time, btime=black_time)
            print(f"original best_move: {best_move}")
        except StockfishException as e:
            print(f"Stockfish error: {e}")
        finally:
            resolved_best_move = self._to_and_from(best_move)
            print(f"resolved_best_move: {resolved_best_move}")

    def _board_to_fen(self, piece_state: np.ndarray) -> str:
        """
        Convert ChessBoardState into a minimal FEN string.
        """
        fen_rows = []

        for row in piece_state:
            empty_count = 0
            fen_row = ""
            for cell in row:
                if cell == "":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    piece_type = cell[1:]
                    if piece_type == "Kn":
                        char = "N"
                    else:
                        char = piece_type
                    char = char.lower() if cell[0] == "b" else char.upper()
                    fen_row += char
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        placement = "/".join(fen_rows)
        active_color = 'b'

        return f"{placement} {active_color} - - 0 1"

    def _to_and_from(self, best_move: str) -> tuple[tuple[int, int], tuple[int, int]]:
        if best_move is not None:
            col_from = ord(best_move[0]) - ord('a')
            row_from = 8 - int(best_move[1])
            col_to = ord(best_move[2]) - ord('a')
            row_to = 8 - int(best_move[3])
            return (col_from, row_from), (col_to, row_to)
