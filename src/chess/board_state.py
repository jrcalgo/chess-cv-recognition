import numpy as np
import pygame


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

    def update_state(self, src, dst):
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

    def promote_piece(self, pos, new_type="Q"):
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
                if self.pieces.piece_state[7, 1] == "" and self.pieces.piece_state[7, 2] == "" and self.pieces.piece_state[7, 3] == "":
                    self.pieces.piece_state[7, 2] = "wK"
                    self.pieces.piece_state[7, 4] = ""
                    self.pieces.piece_state[7, 3] = "wR"
                    self.pieces.piece_state[7, 0] = ""
                    self.current_turn = 'b'
                    return True
        else:
            # Black king must be at (0,4) and rook at (0,0)
            if self.pieces.piece_state[0, 4] == "bK" and self.pieces.piece_state[0, 0] == "bR":
                if self.pieces.piece_state[0, 1] == "" and self.pieces.piece_state[0, 2] == "" and self.pieces.piece_state[0, 3] == "":
                    self.pieces.piece_state[0, 2] = "bK"
                    self.pieces.piece_state[0, 4] = ""
                    self.pieces.piece_state[0, 3] = "bR"
                    self.pieces.piece_state[0, 0] = ""
                    self.current_turn = 'w'
                    return True
        return False

    def en_passant(self, src, dst):
        """
        Perform en passant capture if applicable.
        src: starting position (row, col) of the pawn making the capture.
        dst: destination position (row, col) where the pawn lands.
        """
        piece = self.pieces.piece_state[src[0], src[1]]
        if piece == "" or piece[1] != "P":
            return False
        # Determine direction based on pawn color.
        direction = -1 if piece[0] == 'w' else 1
        # Check if the move is a diagonal step into an empty square.
        if abs(dst[1] - src[1]) == 1 and (dst[0] - src[0]) == direction and self.pieces.piece_state[dst[0], dst[1]] == "":
            # The pawn to capture is directly adjacent in the same row as src, in column dst[1].
            captured = self.pieces.piece_state[src[0], dst[1]]
            if captured != "" and captured[1] == "P" and captured[0] != piece[0]:
                # Additionally, require that the captured pawn moved two steps last turn.
                if self.last_move and self.last_move[0] == captured:
                    src_captured = self.last_move[1]
                    dst_captured = self.last_move[2]
                    if abs(src_captured[0] - dst_captured[0]) == 2:
                        # Perform en passant: move pawn and remove captured pawn.
                        self.pieces.piece_state[dst[0], dst[1]] = piece
                        self.pieces.piece_state[src[0], src[1]] = ""
                        self.pieces.piece_state[src[0], dst[1]] = ""
                        self.last_move = (piece, src, dst)
                        self.current_turn = 'b' if self.current_turn == 'w' else 'w'
                        return True
        return False

    def check_for_checkmate(self):
        """
        Dummy checkmate: if one of the kings is missing.
        (A full checkmate check would require move generation and check validation.)
        """
        kings = 0
        for i in range(8):
            for j in range(8):
                if self.pieces.piece_state[i, j] in ["wK", "bK"]:
                    kings += 1
        return kings < 2

    def check_for_stalemate(self):
        """
        Dummy stalemate: not fully implemented.
        """
        return False


class ChessGame:
    def __init__(self):
        self.board_state = ChessBoardState()
        self.selected = None  # For selecting a piece to move (source square).
        self.square_size = 100

    def run_game(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption('Chess')
        font = pygame.font.SysFont(None, 40)
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = x // self.square_size
                    row = y // self.square_size

                    if self.selected is None:
                        # Select a piece if the square is not empty.
                        if self.board_state.pieces.piece_state[row, col] != "":
                            self.selected = (row, col)
                    else:
                        # Try to move the selected piece to the new square.
                        moved = self.board_state.update_state(self.selected, (row, col))
                        if not moved:
                            # If a normal move was not valid, try en passant.
                            self.board_state.en_passant(self.selected, (row, col))
                        self.selected = None

            # Draw the board
            for i in range(8):
                for j in range(8):
                    # Use a light color for even tiles and a dark color for odd tiles.
                    tile_color = (238, 238, 210) if (i + j) % 2 == 0 else (118, 150, 86)
                    rect = pygame.Rect(j * self.square_size, i * self.square_size,
                                       self.square_size, self.square_size)
                    pygame.draw.rect(screen, tile_color, rect)

                    # Draw a selection highlight if this square is selected.
                    if self.selected == (i, j):
                        pygame.draw.rect(screen, (200, 0, 0), rect, 3)

                    # Draw any piece that exists in this square.
                    piece: str = str(self.board_state.pieces.piece_state[i, j])
                    if piece != "":
                        text = font.render(piece, True, (0, 0, 0))
                        text_rect = text.get_rect(center=rect.center)
                        screen.blit(text, text_rect)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def get_board_state(self):
        return self.board_state

    def move_piece(self, src, dst):
        self.board_state.update_state(src, dst)

    def promote_piece(self, pos, new_type="Q"):
        self.board_state.promote_piece(pos, new_type)

    def castle_kingside(self):
        self.board_state.castle_kingside()

    def castle_queenside(self):
        self.board_state.castle_queenside()

    def en_passant(self, src, dst):
        self.board_state.en_passant(src, dst)

    def checkmate(self):
        return self.board_state.check_for_checkmate()

    def stalemate(self):
        return self.board_state.check_for_stalemate()


if __name__ == '__main__':
    game = ChessGame()
    game.run_game()
