import pygame
from peice_locating.peice_locating import get_center_point, get_cell, gen_grid, draw_grid
from ..training_scripts.realtime_chess_detection import CVModel


screen = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption("perspective grid gen")
grid = None
corners = []
player_turn = True
move_made = False
cv_model = CVModel("../models/best_v2.pt")
cv_model.start_capture(2)
while True:
    move_made = False
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            corners.append(pygame.Vector2(pygame.mouse.get_pos()))
            if len(corners) == 4:
                grid = gen_grid(corners, 8)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                move_made = True

    if grid == None:
        continue

    piece_boxes = cv_model.get_piece_loc()
    if piece_boxes == None:
        continue

    piece_locations = {}
    for piece in piece_boxes:
        piece_info = piece_boxes[piece]
        width = piece_info[2] - piece_info[0]
        height = piece_info[3] - piece_info[1]
        peice_rect = pygame.Rect(piece_info[0], piece_info[1], width, height)
        base_point = get_center_point(peice_rect)
        piece_locations[piece] = get_cell(base_point, grid)
    print(piece_locations)
    draw_grid(grid, screen)
    if not move_made:
        continue
    if player_turn:
        pass
    else:
        pass
