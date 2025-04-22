import numpy as np
import pygame

letter_map = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

def gen_grid(corners: list[pygame.Vector2], grid_size: int) -> list[list[pygame.Vector2]]:
    src = [(0, 0), (1, 0), (1, 1), (0, 1)]
    H = compute_homography(src, corners)
    grid = []
    for row in range(grid_size + 1):
        row_points = []
        v = row / grid_size
        for col in range(grid_size + 1):
            u = col / grid_size
            point = pygame.Vector2(apply_homography(H, (u, v)))
            row_points.append(point)
        grid.append(row_points)
    return grid

def compute_homography(src_pts, dst_pts):
    A = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src_pts, dst_pts):
        A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst])
        A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_dst, y_src * y_dst, y_dst])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    return L.reshape(3, 3)

def apply_homography(H, pt):
    x, y = pt
    vec = np.array([x, y, 1])
    result = H @ vec
    result /= result[2]
    return (result[0], result[1])

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
    global letter_map
    for row in range(len(grid)-1):
        for col in range(len(grid[row])-1):
            if is_in_cell(point, grid, row, col):
                return letter_map[col] + str(row)
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
    rect = pygame.Rect(230, 290, 40, 80)
    center_of_base = get_center_point(rect)
    screen = pygame.display.set_mode((428, 571))
    pygame.display.set_caption("perspective grid gen")
    img = pygame.image.load('chess_board.jpg')
    img = pygame.transform.smoothscale(img, (428, 571))
    screen.blit(img,(0,0))
    grid = None
    while True:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                corners.append(pygame.Vector2(pygame.mouse.get_pos()))
                if len(corners) == 4:
                    grid = gen_grid(corners, 8)
                    print(get_cell(center_of_base, grid))

        if grid == None:
            continue
        draw_grid(grid, screen)
        pygame.draw.rect(screen, (255,255,0), rect)
        pygame.draw.circle(screen, (0, 255, 255), center_of_base, 3)
