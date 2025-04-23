from stockfish import Stockfish, StockfishException

from src.chess.board_state import ChessBoardState


class StockfishPlayer:
    def __init__(self, chess_board):
        self.current_board: ChessBoardState = chess_board
        self.stockfish = Stockfish()

    def get_stockfish_move(self, recent_chess_board: ChessBoardState, white_time: int, black_time: int) -> str:
        try:
            # Convert ChessBoardState to FEN
            fen = self._board_to_fen(recent_chess_board)
            # Update Stockfish with the current position
            self.stockfish.set_fen_position(fen)
            # Retrieve and return the best move
            best_move = self.stockfish.get_best_move(wtime=white_time, btime=black_time)
        except StockfishException as e:
            print(f"Stockfish error: {e}")
            best_move = None
        finally:
            return best_move

    def _board_to_fen(self, chess_board: ChessBoardState) -> str:
        """
        Convert ChessBoardState into a minimal FEN string.
        """
        piece_state = chess_board.pieces.piece_state
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
        active_color = chess_board.current_turn  # 'w' or 'b'

        return f"{placement} {active_color} - - 0 1"
