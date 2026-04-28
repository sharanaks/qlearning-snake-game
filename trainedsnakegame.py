import pygame
import random
import numpy as np
import pickle

pygame.init()

WIDTH, HEIGHT = 400, 500
BLOCK = 20

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake AI (Q-Learning Training) 🧠🐍")

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
mode = "AI"   # start directly in AI for training

# Food
def new_food():
    return (
        random.randint(0, (WIDTH//BLOCK)-1) * BLOCK,
        random.randint(0, (400//BLOCK)-1) * BLOCK
    )

food = new_food()

# Only START button kept
start_btn = pygame.Rect(120, 200, 160, 50)

# -------- Q-LEARNING --------

# Load saved model
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
    screen.blit(font.render("Snake AI Training 🐍", True, GREEN), (90,120))
    draw_button(start_btn, "START")
    pygame.display.update()

# -------- RESET --------
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

    else:
        clock.tick(50)  # 🔥 faster training

        # AI always active
        state = get_state()
        action = choose_action(state)
        direction = actions[action]

        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        reward = -0.1

        # collision
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
        next_state = get_state()

        if state not in Q:
            Q[state] = [0,0,0,0]
        if next_state not in Q:
            Q[next_state] = [0,0,0,0]

        Q[state][action] += alpha * (
            reward + gamma * max(Q[next_state]) - Q[state][action]
        )

        # 🔁 AUTO RESET (IMPORTANT)
        if game_over:
            reset()

        draw()

    # EVENTS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            with open("qtable.pkl", "wb") as f:
                pickle.dump(Q, f)
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if start_btn.collidepoint(event.pos):
                game_started = True

pygame.quit()