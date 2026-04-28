import pygame
import random
import numpy as np
import pickle   # ✅ ADDED

pygame.init()

WIDTH, HEIGHT = 400, 500
BLOCK = 20

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake AI (Q-Learning) 🧠🐍")

clock = pygame.time.Clock()

# Colors
GREEN = (0,200,0)
LIGHT_GREEN = (200,255,200)
RED = (255,0,0)
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (150,150,255)

# Game state
snake = [(100,100)]
direction = (BLOCK,0)
score = 0
game_over = False
game_started = False
mode = "HUMAN"

# Food
def new_food():
    return (
        random.randint(0, (WIDTH//BLOCK)-1) * BLOCK,
        random.randint(0, (400//BLOCK)-1) * BLOCK
    )

food = new_food()

# Buttons
start_btn = pygame.Rect(120, 200, 160, 50)
restart_btn = pygame.Rect(120, 260, 160, 50)

# -------- Q-LEARNING --------

# ✅ LOAD SAVED MODEL (if exists)
try:
    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)
except:
    Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

actions = [(0,-BLOCK),(0,BLOCK),(-BLOCK,0),(BLOCK,0)]

def get_state():
    head = snake[0]
    dx = food[0] - head[0]
    dy = food[1] - head[1]

    return (
        int(dx > 0),
        int(dx < 0),
        int(dy > 0),
        int(dy < 0)
    )

def choose_action(state):
    if state not in Q:
        Q[state] = [0,0,0,0]

    if random.random() < epsilon:
        return random.randint(0,3)

    return np.argmax(Q[state])

# -------- DRAW --------
def draw_button(rect, text):
    font = pygame.font.SysFont(None, 35)
    pygame.draw.rect(screen, BLUE, rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, rect, 2, border_radius=10)
    txt = font.render(text, True, BLACK)
    screen.blit(txt, txt.get_rect(center=rect.center))

def draw():
    screen.fill(LIGHT_GREEN)

    for i, s in enumerate(snake):
        pygame.draw.rect(screen, GREEN, (*s, BLOCK, BLOCK), border_radius=8)
        if i == 0:
            pygame.draw.circle(screen, BLACK, (s[0]+6, s[1]+6), 2)
            pygame.draw.circle(screen, BLACK, (s[0]+14, s[1]+6), 2)

    pygame.draw.circle(screen, RED, (food[0]+BLOCK//2, food[1]+BLOCK//2), BLOCK//2-2)

    font = pygame.font.SysFont(None, 30)
    screen.blit(font.render(f"Score: {score}", True, BLACK), (10,10))
    screen.blit(font.render(f"Mode: {mode}", True, BLACK), (250,10))

    pygame.display.update()

def show_start():
    screen.fill(WHITE)
    font = pygame.font.SysFont(None, 40)
    screen.blit(font.render("Snake Game 🐍", True, GREEN), (110,120))
    draw_button(start_btn, "START")
    pygame.display.update()

def show_game_over():
    screen.fill(WHITE)
    font = pygame.font.SysFont(None, 40)
    screen.blit(font.render("Game Over!", True, RED), (120,150))
    screen.blit(font.render(f"Score: {score}", True, BLACK), (140,190))
    draw_button(restart_btn, "RESTART")
    pygame.display.update()

def reset():
    global snake, direction, score, food, game_over
    snake = [(100,100)]
    direction = (BLOCK,0)
    score = 0
    food = new_food()
    game_over = False

# -------- MAIN LOOP --------
running = True

while running:

    if not game_started:
        show_start()

    elif game_over:
        show_game_over()

    else:
        clock.tick(5)

        # AI
        if mode == "AI":
            state = get_state()
            action = choose_action(state)
            direction = actions[action]

        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        reward = -0.1

        if (head[0] < 0 or head[0] >= WIDTH or
            head[1] < 0 or head[1] >= 400 or
            head in snake[1:]):
            reward = -10
            game_over = True

        snake.insert(0, head)

        if head == food:
            reward = 10
            score += 1
            food = new_food()
        else:
            snake.pop()

        # Q-learning update
        if mode == "AI":
            next_state = get_state()

            if state not in Q:
                Q[state] = [0,0,0,0]
            if next_state not in Q:
                Q[next_state] = [0,0,0,0]

            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state]) - Q[state][action]
            )

        draw()

    # EVENTS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # ✅ SAVE MODEL BEFORE EXIT
            with open("qtable.pkl", "wb") as f:
                pickle.dump(Q, f)

            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            if start_btn.collidepoint(pos):
                game_started = True

            if restart_btn.collidepoint(pos):
                reset()
                game_started = True

        if event.type == pygame.KEYDOWN:

            if mode == "HUMAN":
                if event.key == pygame.K_UP and direction != (0,BLOCK):
                    direction = (0,-BLOCK)
                elif event.key == pygame.K_DOWN and direction != (0,-BLOCK):
                    direction = (0,BLOCK)
                elif event.key == pygame.K_LEFT and direction != (BLOCK,0):
                    direction = (-BLOCK,0)
                elif event.key == pygame.K_RIGHT and direction != (-BLOCK,0):
                    direction = (BLOCK,0)

            if event.key == pygame.K_a:
                mode = "AI"
            elif event.key == pygame.K_h:
                mode = "HUMAN"

pygame.quit()