import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import sys
import time
import os
from pygame.locals import *

# Thiết lập MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Các màu sắc cải tiến - màu sắc rực rỡ hơn với hiệu ứng gradient
colors = [
    (0, 0, 0),          # Đen - nền
    (0, 240, 240),      # Xanh lam - I
    (240, 160, 0),      # Cam - L
    (0, 120, 240),      # Xanh dương - J
    (240, 240, 0),      # Vàng - O
    (0, 240, 0),        # Xanh lá - S
    (160, 0, 240),      # Tím - T
    (240, 0, 0)         # Đỏ - Z
]

# Màu sắc thứ cấp cho hiệu ứng gradient
secondary_colors = [
    (0, 0, 0),          # Đen - nền
    (0, 200, 200),      # Xanh lam - I
    (200, 120, 0),      # Cam - L
    (0, 80, 200),       # Xanh dương - J
    (200, 200, 0),      # Vàng - O
    (0, 200, 0),        # Xanh lá - S
    (120, 0, 200),      # Tím - T
    (200, 0, 0)         # Đỏ - Z
]

# Các khối Tetris
tetris_shapes = [
    [[1, 1, 1, 1]],                       # I
    [[2, 0, 0], [2, 2, 2]],               # L
    [[0, 0, 3], [3, 3, 3]],               # J
    [[4, 4], [4, 4]],                     # O
    [[0, 5, 5], [5, 5, 0]],               # S
    [[0, 6, 0], [6, 6, 6]],               # T
    [[7, 7, 0], [0, 7, 7]]                # Z
]

# Thiết lập cửa sổ Pygame
pygame.init()
pygame.display.set_caption('Tetris')

# Sử dụng font hệ thống để đảm bảo tiếng Việt không bị lỗi
main_font = pygame.font.Font("game/fonts/Roboto-Regular.ttf", 16)
guide_font = pygame.font.Font("game/fonts/Roboto-Regular.ttf", 15)
title_font = pygame.font.Font("game/fonts/Roboto-Regular.ttf", 40)

# Kích thước ô vuông
cell_size = 30
cols = 10
rows = 20

# Thiết lập kích thước và giao diện
status_width = 280  # Khu vực trạng thái bên phải rộng hơn
window_width = cols * cell_size + status_width
window_height = rows * cell_size

# Thiết lập màu nền
bg_color = (25, 25, 35)  # Màu xanh đậm cho nền
grid_color = (40, 40, 60)  # Màu lưới
border_color = (100, 100, 140)  # Màu viền

# Tạo khung hình
screen = pygame.display.set_mode((window_width, window_height))
game_area = pygame.Surface((cols * cell_size, rows * cell_size))

# Các biến thời gian
clock = pygame.time.Clock()
fps = 60  # Tăng fps để animation mượt hơn

# Biến trạng thái
score = 0
level = 1
lines_cleared = 0
game_over = False
paused = False

# Hiệu ứng
particles = []
flash_effect = 0
animation_speed = 5

# Tốc độ rơi khối
fall_speed = 1.0  # Số giây để rơi xuống một hàng
last_fall_time = time.time()

# Biến dành cho Tetris
grid = [[0 for _ in range(cols)] for _ in range(rows)]
current_piece = None
next_piece = None
hold_piece = None
can_hold = True  # Biến để kiểm soát việc giữ khối

# Điểm cho khối rơi
ghost_piece_position = 0

# Âm thanh
def load_sound(filename):
    path = os.path.join('game/sounds', filename)
    if os.path.isfile(path):
        return pygame.mixer.Sound(path)
    else:
        print(f"Không tìm thấy file âm thanh: {filename}")
        return None

try:
    pygame.mixer.init()
    rotate_sound = load_sound('rotate.wav')
    clear_sound = load_sound('clear.wav')
    drop_sound = load_sound('drop.wav')
    game_over_sound = load_sound('gameover.wav')
    # Điều chỉnh âm lượng (1.0 là to nhất)
    if rotate_sound: rotate_sound.set_volume(1.0)
    if clear_sound: clear_sound.set_volume(1.0)
    if drop_sound: drop_sound.set_volume(1.0)
    if game_over_sound: game_over_sound.set_volume(1.0)
except Exception as e:
    print("Lỗi khởi tạo âm thanh:", e)
    rotate_sound = None
    clear_sound = None
    drop_sound = None
    game_over_sound = None

def new_piece():
    global next_piece
    if next_piece is None:
        shape = random.choice(tetris_shapes)
    else:
        shape = next_piece
    next_piece = random.choice(tetris_shapes)
    # Vị trí bắt đầu
    x = cols // 2 - len(shape[0]) // 2
    y = 0
    return {'shape': shape, 'x': x, 'y': y, 'rotation': 0}

def rotate_piece(piece):
    # Xoay khối 90 độ theo chiều kim đồng hồ
    shape = piece['shape']
    rotated = [[] for _ in range(len(shape[0]))]
    for i in range(len(shape)):
        for j in range(len(shape[0])):
            rotated[j].append(shape[len(shape) - 1 - i][j])
    # Phát âm thanh xoay nếu có
    if rotate_sound:
        rotate_sound.stop()
        rotate_sound.play()
    return rotated

def is_valid_position(piece, grid, x_offset=0, y_offset=0):
    if piece is None:
        return False
    shape = piece['shape']
    for y in range(len(shape)):
        for x in range(len(shape[0])):
            if shape[y][x] != 0:  # Nếu ô không phải là ô trống
                new_x = piece['x'] + x + x_offset
                new_y = piece['y'] + y + y_offset
                # Kiểm tra xem ô có nằm ngoài lưới không
                if new_x < 0 or new_x >= cols or new_y >= rows:
                    return False
                # Kiểm tra xung đột với các khối đã đặt
                if new_y >= 0 and grid[new_y][new_x] != 0:
                    return False
    return True

def merge_piece_with_grid(piece, grid):
    if piece is None:
        return
    shape = piece['shape']
    for y in range(len(shape)):
        for x in range(len(shape[0])):
            if shape[y][x] != 0:
                grid[piece['y'] + y][piece['x'] + x] = shape[y][x]
    # Phát âm thanh khi thả khối
    if drop_sound:
        drop_sound.stop()
        drop_sound.play()

def clear_rows(grid):
    global flash_effect, particles
    full_rows = []
    for i in range(rows):
        if all(cell != 0 for cell in grid[i]):
            full_rows.append(i)
    if full_rows:
        flash_effect = 10  # Tạo hiệu ứng nhấp nháy
        # Tạo hiệu ứng particle khi xóa hàng
        for row in full_rows:
            for _ in range(20):  # 20 particles per row
                particles.append({
                    'x': random.randint(0, cols * cell_size),
                    'y': row * cell_size,
                    'dx': random.uniform(-2, 2),
                    'dy': random.uniform(-3, -1),
                    'color': random.choice([(255,255,255), (200,200,0), (0,200,200)]),
                    'life': 60
                })
        # Phát âm thanh khi xóa hàng
        if clear_sound:
            clear_sound.stop()
            clear_sound.play()
    # Xóa hàng đầy và thêm hàng trống ở trên cùng
    for row in full_rows:
        del grid[row]
        grid.insert(0, [0 for _ in range(cols)])
    return len(full_rows)

def update_ghost_piece(piece, grid):
    """Cập nhật vị trí của ghost piece (đánh dấu nơi khối sẽ rơi)"""
    if piece is None:
        return 0
    ghost_y = 0
    while is_valid_position(piece, grid, y_offset=ghost_y + 1):
        ghost_y += 1
    return ghost_y

def update_particles():
    global particles
    for particle in particles[:]:
        particle['x'] += particle['dx']
        particle['y'] += particle['dy']
        particle['life'] -= 1
        if particle['life'] <= 0:
            particles.remove(particle)

def draw_grid(surface, grid):
    pygame.draw.rect(surface, bg_color, (0, 0, cols * cell_size, rows * cell_size))
    for y in range(rows):
        for x in range(cols):
            pygame.draw.rect(
                surface, 
                grid_color, 
                (x * cell_size, y * cell_size, cell_size, cell_size), 
                1
            )
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] != 0:
                color_idx = grid[y][x]
                pygame.draw.rect(
                    surface, 
                    secondary_colors[color_idx], 
                    (x * cell_size, y * cell_size, cell_size, cell_size)
                )
                pygame.draw.rect(
                    surface, 
                    colors[color_idx], 
                    (x * cell_size + 3, y * cell_size + 3, cell_size - 6, cell_size - 6)
                )
                pygame.draw.rect(
                    surface, 
                    (255, 255, 255, 128), 
                    (x * cell_size, y * cell_size, cell_size, cell_size), 
                    1
                )

def draw_piece(surface, piece, ghost=False, offset_y=0):
    if piece is None:
        return
    shape = piece['shape']
    for y in range(len(shape)):
        for x in range(len(shape[0])):
            if shape[y][x] != 0:
                color_idx = shape[y][x]
                pos_x = (piece['x'] + x) * cell_size
                pos_y = (piece['y'] + y + offset_y) * cell_size
                if ghost:
                    pygame.draw.rect(
                        surface, 
                        (colors[color_idx][0]//3, colors[color_idx][1]//3, colors[color_idx][2]//3, 128), 
                        (pos_x, pos_y, cell_size, cell_size), 
                        1
                    )
                    pygame.draw.rect(
                        surface, 
                        (colors[color_idx][0]//4, colors[color_idx][1]//4, colors[color_idx][2]//4, 64), 
                        (pos_x + 3, pos_y + 3, cell_size - 6, cell_size - 6), 
                        1
                    )
                else:
                    pygame.draw.rect(
                        surface, 
                        secondary_colors[color_idx], 
                        (pos_x, pos_y, cell_size, cell_size)
                    )
                    pygame.draw.rect(
                        surface, 
                        colors[color_idx], 
                        (pos_x + 3, pos_y + 3, cell_size - 6, cell_size - 6)
                    )
                    pygame.draw.rect(
                        surface, 
                        (255, 255, 255, 128), 
                        (pos_x, pos_y, cell_size, cell_size), 
                        1
                    )

def draw_piece_preview(surface, shape, x, y, size=20):
    if shape is None:
        return
    width = len(shape[0]) * size
    height = len(shape) * size
    start_x = x + (4 * size - width) // 2
    start_y = y + (3 * size - height) // 2
    for row_idx in range(len(shape)):
        for col_idx in range(len(shape[0])):
            if shape[row_idx][col_idx] != 0:
                color_idx = shape[row_idx][col_idx]
                pygame.draw.rect(
                    surface, 
                    secondary_colors[color_idx], 
                    (start_x + col_idx * size, start_y + row_idx * size, size, size)
                )
                pygame.draw.rect(
                    surface, 
                    colors[color_idx], 
                    (start_x + col_idx * size + 2, start_y + row_idx * size + 2, size - 4, size - 4)
                )
                pygame.draw.rect(
                    surface, 
                    (255, 255, 255, 128), 
                    (start_x + col_idx * size, start_y + row_idx * size, size, size), 
                    1
                )

def draw_next_piece(surface, piece):
    pygame.draw.rect(
        surface, 
        border_color, 
        (cols * cell_size + 20, 10, 4 * 20 + 20, 3 * 20 + 20), 
        2, 
        5
    )
    text = main_font.render("Khối tiếp theo", True, (220, 220, 220))
    surface.blit(text, (cols * cell_size + 25, 12))
    draw_piece_preview(surface, piece, cols * cell_size + 30, 32)

def draw_hold_piece(surface, piece):
    pygame.draw.rect(
        surface, 
        border_color, 
        (cols * cell_size + 20, 70, 4 * 20 + 20, 3 * 20 + 20), 
        2, 
        5
    )
    text = main_font.render("Giữ", True, (220, 220, 220))
    surface.blit(text, (cols * cell_size + 25, 72))
    if piece:
        draw_piece_preview(surface, piece, cols * cell_size + 30, 92)
    else:
        text = guide_font.render("Trống", True, (150, 150, 150))
        surface.blit(text, (cols * cell_size + 50, 100))

def draw_particles(surface):
    for particle in particles:
        color = particle['color']
        alpha = min(255, int(particle['life'] * 4))
        pygame.draw.circle(
            surface, 
            (color[0], color[1], color[2], alpha),
            (int(particle['x']), int(particle['y'])), 
            max(1, particle['life'] // 10)
        )

def draw_status(surface):
    # Nền panel phải (status)
    pygame.draw.rect(
        surface,
        (20, 22, 35),  # Đậm hơn 1 chút
        (cols * cell_size, 0, status_width, rows * cell_size)
    )
    pygame.draw.line(
        surface,
        border_color,
        (cols * cell_size, 0),
        (cols * cell_size, rows * cell_size),
        3
    )

    # Vùng "Khối tiếp theo" 
    next_box_x = cols * cell_size + 16
    next_box_y = 8
    next_box_w = status_width - 32
    next_box_h = 90  # tăng cao
    pygame.draw.rect(surface, (30, 31, 50), (next_box_x, next_box_y, next_box_w, next_box_h), 0, 8)
    pygame.draw.rect(surface, border_color, (next_box_x, next_box_y, next_box_w, next_box_h), 2, 8)

    # Hiển thị tiêu đề "Khối tiếp theo"
    text = main_font.render("Khối tiếp theo", True, (220, 220, 220))
    surface.blit(text, (next_box_x + 10, next_box_y + 4))

    # Vẽ khối tiếp theo vào giữa box
    # Kích thước mỗi ô của khối
    block_size = 25  # Kích thước mỗi ô (size=25)

    # Lấy kích thước của khối (giả sử khối là ma trận 2D)
    piece_width = len(next_piece[0]) * block_size  # Số ô ngang * kích thước ô
    piece_height = len(next_piece) * block_size   # Số ô dọc * kích thước ô

    # Tính toán vị trí để căn giữa khối trong box
    center_x = next_box_x + (next_box_w - piece_width) // 2
    center_y = next_box_y + (next_box_h - piece_height) // 2

    # Vẽ khối tiếp theo (căn giữa)
    draw_piece_preview(surface, next_piece, center_x, center_y, size=block_size)

    # Vùng "Giữ"
    hold_box_y = next_box_y + next_box_h + 6
    pygame.draw.rect(surface, (30, 31, 50), (next_box_x, hold_box_y, next_box_w, 64), 0, 8)
    pygame.draw.rect(surface, border_color, (next_box_x, hold_box_y, next_box_w, 64), 2, 8)
    text = main_font.render("Giữ", True, (220, 220, 220))
    surface.blit(text, (next_box_x + 10, hold_box_y + 2))
    if hold_piece:
        draw_piece_preview(surface, hold_piece, next_box_x + 18, hold_box_y + 28, size=25)
    else:
        text = guide_font.render("Trống", True, (150, 150, 150))
        surface.blit(text, (next_box_x + 50, hold_box_y + 32))

    # Box điểm
    info_box_y = hold_box_y + 70
    pygame.draw.rect(surface, (35, 38, 56), (next_box_x, info_box_y, next_box_w, 70), 0, 8)
    score_text = main_font.render(f"Điểm: {score}", True, (255, 255, 255))
    surface.blit(score_text, (next_box_x + 12, info_box_y + 8))
    level_text = main_font.render(f"Cấp độ: {level}", True, (255, 255, 255))
    surface.blit(level_text, (next_box_x + 12, info_box_y + 32))
    lines_text = main_font.render(f"Hàng: {lines_cleared}", True, (255, 255, 255))
    surface.blit(lines_text, (next_box_x + 12, info_box_y + 56))

    # Box hướng dẫn
    guide_box_y = info_box_y + 78
    pygame.draw.rect(surface, (35, 38, 56), (next_box_x, guide_box_y, next_box_w, 340), 0, 8)
    guide_title = main_font.render("Hướng dẫn", True, (220, 220, 220))
    surface.blit(guide_title, (next_box_x + 10, guide_box_y + 4))
    guide_text = [
        "Cử chỉ tay:",
        "- Ngón trỏ nghiêng trái: Sang trái",
        "- Ngón trỏ nghiêng phải: Sang phải",
        "- Gập ngón trỏ: Xoay khối",
        "- Hạ thấp cổ tay: Thả nhanh",
        "",
        "Bàn phím:",
        "- Mũi tên: Di chuyển",
        "- Phím SPACE: Thả nhanh",
        "- Phím C: Giữ khối",
        "- Phím P: Tạm dừng",
        "- Phím R: Chơi lại",
        "- Phím ESC: Thoát"
    ]
    for i, line in enumerate(guide_text):
        text = guide_font.render(line, True, (200, 200, 200))
        surface.blit(text, (next_box_x + 12, guide_box_y + 38 + i * 22))

def detect_hand_gesture(frame):
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = {
        'move_left': False,
        'move_right': False,
        'rotate': False,
        'drop': False,
        'hold': False
    }
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

            # Sang trái/phải: dựa vào hướng ngón trỏ
            vector_x = index_tip.x - index_pip.x
            if vector_x < -0.05:
                gesture['move_left'] = True
            elif vector_x > 0.02:
                gesture['move_right'] = True

            # Xoay: gập ngón trỏ lại (tip.y > pip.y)
            if index_tip.y > index_pip.y + 0.02:
                gesture['rotate'] = True

            # Thả nhanh: cổ tay thấp
            if wrist.y > 0.85:
                gesture['drop'] = True

    cv2.imshow('Hand Tracking', frame)
    return gesture

def show_game_over_screen(surface):
    overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))
    pygame.draw.rect(
        surface, 
        (60, 60, 80), 
        (window_width // 2 - 150, window_height // 2 - 100, 300, 200), 
        0, 
        10
    )
    pygame.draw.rect(
        surface, 
        border_color, 
        (window_width // 2 - 150, window_height // 2 - 100, 300, 200), 
        2, 
        10
    )
    game_over_text = title_font.render("GAME OVER", True, (255, 50, 50))
    surface.blit(game_over_text, (window_width // 2 - game_over_text.get_width() // 2, window_height // 2 - 80))
    score_text = main_font.render(f"Điểm: {score}", True, (220, 220, 220))
    surface.blit(score_text, (window_width // 2 - score_text.get_width() // 2, window_height // 2 - 20))
    lines_text = main_font.render(f"Số hàng: {lines_cleared}", True, (220, 220, 220))
    surface.blit(lines_text, (window_width // 2 - lines_text.get_width() // 2, window_height // 2 + 10))
    restart_text = main_font.render("Nhấn R để chơi lại", True, (240, 240, 240))
    surface.blit(restart_text, (window_width // 2 - restart_text.get_width() // 2, window_height // 2 + 50))

def show_pause_screen(surface):
    overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))
    pygame.draw.rect(
        surface, 
        (60, 60, 80), 
        (window_width // 2 - 120, window_height // 2 - 60, 240, 120), 
        0, 
        10
    )
    pygame.draw.rect(
        surface, 
        border_color, 
        (window_width // 2 - 120, window_height // 2 - 60, 240, 120), 
        2, 
        10
    )
    pause_text = title_font.render("TẠM DỪNG", True, (220, 220, 220))
    surface.blit(pause_text, (window_width // 2 - pause_text.get_width() // 2, window_height // 2 - 50))
    resume_text = main_font.render("Nhấn P để tiếp tục", True, (240, 240, 240))
    surface.blit(resume_text, (window_width // 2 - resume_text.get_width() // 2, window_height // 2 + 20))

def hold_current_piece():
    global current_piece, hold_piece, can_hold
    if not can_hold:
        return
    if hold_piece is None:
        hold_piece = current_piece['shape']
        current_piece = new_piece()
    else:
        temp = hold_piece
        hold_piece = current_piece['shape']
        current_piece = {'shape': temp, 'x': cols // 2 - len(temp[0]) // 2, 'y': 0, 'rotation': 0}
    can_hold = False

current_piece = new_piece()
next_piece = new_piece()['shape']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam!")
    sys.exit()

running = True
gesture_timer = 0
gesture_delay = 0.14

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_p:
                paused = not paused
            elif event.key == K_r:
                grid = [[0 for _ in range(cols)] for _ in range(rows)]
                score = 0
                level = 1
                lines_cleared = 0
                game_over = False
                paused = False
                current_piece = new_piece()
                next_piece = new_piece()['shape']
                hold_piece = None
                can_hold = True
                particles = []
            elif event.key == K_LEFT or event.key == K_a:
                if not paused and not game_over and is_valid_position(current_piece, grid, x_offset=-1):
                    current_piece['x'] -= 1
            elif event.key == K_RIGHT or event.key == K_d:
                if not paused and not game_over and is_valid_position(current_piece, grid, x_offset=1):
                    current_piece['x'] += 1
            elif event.key == K_DOWN or event.key == K_s:
                if not paused and not game_over and is_valid_position(current_piece, grid, y_offset=1):
                    current_piece['y'] += 1
            elif event.key == K_UP or event.key == K_w:
                if not paused and not game_over:
                    rotated_shape = rotate_piece(current_piece)
                    temp_piece = current_piece.copy()
                    temp_piece['shape'] = rotated_shape
                    if is_valid_position(temp_piece, grid):
                        current_piece['shape'] = rotated_shape
            elif event.key == K_SPACE:
                if not paused and not game_over:
                    while is_valid_position(current_piece, grid, y_offset=1):
                        current_piece['y'] += 1
                    merge_piece_with_grid(current_piece, grid)
                    rows_cleared = clear_rows(grid)
                    if rows_cleared > 0:
                        lines_cleared += rows_cleared
                        score += rows_cleared * 100 * level
                        level = lines_cleared // 10 + 1
                        fall_speed = max(0.1, 1.0 - (level - 1) * 0.05)
                    current_piece = new_piece()
                    can_hold = True
                    if not is_valid_position(current_piece, grid):
                        game_over = True
                        if game_over_sound:
                            game_over_sound.stop()
                            game_over_sound.play()
            elif event.key == K_c:
                if not paused and not game_over:
                    hold_current_piece()

    if not paused and not game_over:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ webcam!")
            break

        now = time.time()
        gesture = detect_hand_gesture(frame)
        if now - gesture_timer > gesture_delay:
            gesture_timer = now
            if gesture['move_left'] and is_valid_position(current_piece, grid, x_offset=-1):
                current_piece['x'] -= 1
            elif gesture['move_right'] and is_valid_position(current_piece, grid, x_offset=1):
                current_piece['x'] += 1
            elif gesture['rotate']:
                rotated_shape = rotate_piece(current_piece)
                temp_piece = current_piece.copy()
                temp_piece['shape'] = rotated_shape
                if is_valid_position(temp_piece, grid):
                    current_piece['shape'] = rotated_shape
            elif gesture['drop']:
                while is_valid_position(current_piece, grid, y_offset=1):
                    current_piece['y'] += 1
                merge_piece_with_grid(current_piece, grid)
                rows_cleared = clear_rows(grid)
                if rows_cleared > 0:
                    lines_cleared += rows_cleared
                    score += rows_cleared * 100 * level
                    level = lines_cleared // 10 + 1
                    fall_speed = max(0.1, 1.0 - (level - 1) * 0.05)
                current_piece = new_piece()
                can_hold = True
                if not is_valid_position(current_piece, grid):
                    game_over = True
                    if game_over_sound:
                        game_over_sound.stop()
                        game_over_sound.play()
            elif gesture['hold']:
                hold_current_piece()

        if time.time() - last_fall_time > fall_speed:
            last_fall_time = time.time()
            if is_valid_position(current_piece, grid, y_offset=1):
                current_piece['y'] += 1
            else:
                merge_piece_with_grid(current_piece, grid)
                rows_cleared = clear_rows(grid)
                if rows_cleared > 0:
                    lines_cleared += rows_cleared
                    score += rows_cleared * 100 * level
                    level = lines_cleared // 10 + 1
                    fall_speed = max(0.1, 1.0 - (level - 1) * 0.05)
                current_piece = new_piece()
                can_hold = True
                if not is_valid_position(current_piece, grid):
                    game_over = True
                    if game_over_sound:
                        game_over_sound.stop()
                        game_over_sound.play()

    game_area.fill(bg_color)
    draw_grid(game_area, grid)
    ghost_offset = update_ghost_piece(current_piece, grid)
    draw_piece(game_area, current_piece, ghost=True, offset_y=ghost_offset)
    draw_piece(game_area, current_piece)
    draw_particles(game_area)
    update_particles()
    
    if flash_effect > 0:
        overlay = pygame.Surface((cols * cell_size, rows * cell_size), pygame.SRCALPHA)
        overlay.fill((255, 255, 100, min(120, flash_effect * 8)))
        game_area.blit(overlay, (0, 0))
        flash_effect -= 1

    screen.fill(bg_color)
    screen.blit(game_area, (0, 0))
    draw_status(screen)

    if paused:
        show_pause_screen(screen)
    if game_over:
        show_game_over_screen(screen)

    pygame.display.flip()
    clock.tick(fps)

    if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()