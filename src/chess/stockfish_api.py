from stockfish import Stockfish, StockfishException

from src.chess.board_state import ChessBoardState


class StockfishPlayer:
    def __init__(self, chess_board):
        self.current_board: ChessBoardState = chess_board
        self.stockfish = Stockfish()

    def get_stockfish_move(self, recent_chess_board: ChessBoardState):
        self.stockfish.get_best_move()

