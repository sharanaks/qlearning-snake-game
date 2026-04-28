# 🐍 Snake ML Agent(Game) using Q-Learning

A Machine Learning based Snake Game AI developed using Reinforcement Learning (Q-Learning).

This project demonstrates how an AI agent learns optimal actions using rewards and penalties instead of fixed instructions.

---
## 🚀 Game Link

👉 https://sharanaks.github.io/qlearning-snake-game/


# 📌 Project Overview

This project implements a self-learning Snake AI using the Q-Learning algorithm.

The ML agent:

* Learns from experience
* Collects rewards for eating food
* Receives penalties for collisions
* Improves gameplay performance over time

The project also includes:

* Human playable mode
* ML training mode
* Saved Q-table model
* Browser-based web version

---

# 🧠 Machine Learning Concept

Machine Learning → Reinforcement Learning → Q-Learning

The Snake acts as an AI Agent interacting with the environment.

---

# 📐 Q-Learning Formula

Q(s,a) = Q(s,a) + α [ R + γ max Q(s',a') - Q(s,a) ]

Where:

| Symbol | Meaning         |
| ------ | --------------- |
| Q(s,a) | Current Q-value |
| α      | Learning rate   |
| γ      | Discount factor |
| R      | Reward          |
| s'     | Next state      |

---

# 🎮 Features

## Desktop Version (Python + pygame)

* Snake ML Agent using Q-Learning
* Human and ML modes
* Continuous training
* Score tracking
* Collision detection
* Saved trained model (qtable.pkl)
* Auto-learning system

## 🌐 Web Version

* Browser playable Snake game
* Human mode
* ML mode
* Swipe controls
* Restart functionality
* Mobile-friendly gameplay

---

# 🛠 Technologies Used

## Programming Languages

* Python
* JavaScript
* HTML
* CSS

## Libraries

* pygame
* numpy
* pickle

---

# 📂 Project Structure

Snake-ML/
│
├── snake_ai.py
├── qtable.pkl
├── index.html
├── README.md

---

# 🚀 How to Run

## Python Version

### Install dependencies

```bash
pip install pygame numpy
```

### Run project

```bash
python snake_ai.py
```

---

## Web Version

Open:

```text
index.html
```

in any browser.

---

# 🧪 Training the AI

The AI improves through repeated gameplay.

Training process:

1. Observe current state
2. Choose action
3. Receive reward or penalty
4. Update Q-values
5. Repeat continuously

The trained model is stored in:

```text
qtable.pkl
```

---

# 📊 Before vs After Training

## Before Training

* Random movement
* Frequent collisions
* Poor performance

## After Training

* Better food collection
* Improved navigation
* Fewer collisions
* Higher scores

---

# 🔥 Future Improvements

Possible future enhancements:

* Deep Q Network (DQN)
* Advanced obstacle avoidance
* Multiplayer mode
* Sound effects
* Better graphics
* Mobile application deployment

---

# 🎓 Educational Purpose

This project was developed to demonstrate:

* Reinforcement Learning
* Q-Learning
* AI Agent behavior
* Game AI implementation
* Practical Machine Learning concepts

---

# 👨‍💻 Author

Sharan

---

# 📜 License

This project is open-source and available for educational purposes.
