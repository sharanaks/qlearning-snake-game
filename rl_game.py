import pygame
import numpy as np
import random

pygame.init()

# Grid
ROWS, COLS = 6, 6
BLOCK = 80

WIDTH, HEIGHT = COLS * BLOCK, ROWS * BLOCK
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Robot (Human + AI)")

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

clock = pygame.time.Clock()

# Q-table
Q = np.zeros((ROWS, COLS, 4))
actions = [(-1,0),(1,0),(0,-1),(0,1)]

# Game elements
start = (0,0)
goal = (5,5)
obstacles = [(1,1),(2,2),(3,1),(4,3)]

# RL params
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Mode
mode = "AI"
state = start

# Score
score = 0
steps = 0
optimal_steps = abs(goal[0]-start[0]) + abs(goal[1]-start[1])

def reset():
    global state, score, steps
    state = start
    score = 0
    steps = 0

def move(s, action):
    x,y = s
    dx,dy = actions[action]
    nx, ny = x+dx, y+dy

    nx = max(0, min(ROWS-1, nx))
    ny = max(0, min(COLS-1, ny))
    return (nx, ny)

def draw(agent):
    screen.fill(WHITE)

    for x in range(ROWS):
        for y in range(COLS):
            rect = pygame.Rect(y*BLOCK, x*BLOCK, BLOCK, BLOCK)

            if (x,y) == goal:
                pygame.draw.rect(screen, GREEN, rect)
            elif (x,y) in obstacles:
                pygame.draw.rect(screen, BLACK, rect)
            elif (x,y) == agent:
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.rect(screen, (200,200,200), rect, 1)

    font = pygame.font.SysFont(None, 28)

    screen.blit(font.render(f"Mode: {mode}", True, (0,0,0)), (10,10))
    screen.blit(font.render(f"Score: {score}", True, (0,0,0)), (10,40))
    screen.blit(font.render(f"Steps: {steps}", True, (0,0,0)), (10,70))

    pygame.display.update()

# -------- TRAIN --------
for episode in range(500):
    s = start

    while s != goal:
        x,y = s

        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[x,y])

        ns = move(s, action)

        if ns == goal:
            r = 10
        elif ns in obstacles:
            r = -10
        else:
            r = -1

        nx, ny = ns
        Q[x,y,action] += alpha * (r + gamma*np.max(Q[nx,ny]) - Q[x,y,action])

        s = ns

        if s in obstacles:
            break

# -------- GAME LOOP --------
running = True
while running:
    clock.tick(5)

    prev_state = state
    moved = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_h:
                mode = "HUMAN"

            elif event.key == pygame.K_a:
                mode = "AI"

            elif event.key == pygame.K_r:
                reset()

            # HUMAN MOVE
            if mode == "HUMAN":
                if event.key == pygame.K_UP:
                    state = move(state, 0); moved = True
                elif event.key == pygame.K_DOWN:
                    state = move(state, 1); moved = True
                elif event.key == pygame.K_LEFT:
                    state = move(state, 2); moved = True
                elif event.key == pygame.K_RIGHT:
                    state = move(state, 3); moved = True

    # AI MOVE
    if mode == "AI":
        x,y = state
        action = np.argmax(Q[x,y])
        state = move(state, action)
        moved = True

    # APPLY REWARD (ONLY ON MOVE + STATE CHANGE)
    if moved and state != prev_state:

        # GOAL
        if state == goal:
            score += 10
            steps += 1

            print("GOAL +10 ✅")

            if steps == optimal_steps:
                print("BEST PATH ✅")
            else:
                print("NOT OPTIMAL ❌")

            pygame.time.delay(1000)
            reset()
            continue

        # OBSTACLE
        elif state in obstacles:
            score -= 10
            steps += 1

            print("HIT OBSTACLE ❌")

            pygame.time.delay(1000)
            reset()
            continue

        # NORMAL STEP
        else:
            score -= 1
            steps += 1

    draw(state)

pygame.quit()