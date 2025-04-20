import numpy as np
import pygame
from pygame.math import Vector2

def gen_grid(corners: list[pygame.Vector2], n_rows: int, n_cols) -> list[list[pygame.Vector2]]:
    grid = []
    for y in range(n_rows):
        grid.append([])
        lerp_perc_r = y / (n_rows - 1)
        p1 = pygame.Vector2.lerp(corners[0], corners[2], lerp_perc_r)
        p2 = pygame.Vector2.lerp(corners[1], corners[3], lerp_perc_r)
        for x in range(n_cols):
            lerp_perc_c = x / (n_cols - 1)
            point = (pygame.Vector2.lerp(p1, p2, lerp_perc_c))
            point = (point * 400) + (100,100)
            grid[y].append(point)
    return grid

def draw_grid(grid: list[list[pygame.Vector2]], surf: pygame.Surface):
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            point = grid[y][x]
            pygame.draw.circle(surf, (255,255,255), point, 5)

def get_center_point(rect: pygame.Rect):
    x = (rect.left + rect.left + rect.width) / 2
    y = ((rect.top + rect.top + rect.height) / 2) + (rect.height / 4)
    return pygame.Vector2(x, y)

def get_cell(point: pygame.Vector2, grid: list[list[pygame.Vector2]]):
    for row in range(len(grid)-1):
        for col in range(len(grid[row])-1):
            if is_in_cell(point, grid, row, col):
                return (row, col)
    return (-1, -1)

def is_in_cell(point: pygame.Vector2, grid: list[list[pygame.Vector2]], row: int, col: int):
    if point.y < ((grid[row][col] + grid[row][col+1]) / 2).y:
        return False
    if point.y > ((grid[row+1][col] + grid[row+1][col+1]) / 2).y:
        return False
    if point.x < ((grid[row][col] + grid[row+1][col]) / 2).x:
        return False
    if point.x > ((grid[row][col+1] + grid[row+1][col+1]) / 2).x:
        return False
    return True

if __name__ == "__main__":
    corners = []
    corners.append(pygame.Vector2(.20,.20))
    corners.append(pygame.Vector2(.80,.20))
    corners.append(pygame.Vector2(0,1))
    corners.append(pygame.Vector2(1,1))
    rect = pygame.Rect(220, 280, 40, 80)
    center_of_base = get_center_point(rect)
    grid = gen_grid(corners, 8, 8)
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("perspective grid gen")
    print(get_cell(center_of_base, grid))
    while True:
        draw_grid(grid, screen)
        pygame.draw.rect(screen, (255,255,0), rect)
        pygame.draw.circle(screen, (0, 255, 255), center_of_base, 3)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
