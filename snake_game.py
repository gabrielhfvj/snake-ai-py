import pygame
import random
import numpy as np

class SnakeGame:
    def __init__(self):
        # Inicialização do jogo
        self.WIDTH = 20
        self.HEIGHT = 20
        self.SNAKE_SIZE = 1
        self.reset()

    def reset(self):
        # Reset do estado do jogo
        self.snake_pos = [(1, 1), (1, 2), (1, 3)]
        self.snake_dir = (0, 0)
        self.apple_pos = (random.randint(0, self.WIDTH - 1), random.randint(0, self.HEIGHT - 1))
        self.done = False
        self.score = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.WIDTH, self.HEIGHT))
        for (x, y) in self.snake_pos:
            state[x, y] = 1
        state[self.apple_pos[0], self.apple_pos[1]] = 2  # Representa a maçã
        return state

    def step(self, action):
        # Atualiza o estado do jogo com base na ação
        if action == 0:  # UP
            self.snake_dir = (0, -1)
        elif action == 1:  # DOWN
            self.snake_dir = (0, 1)
        elif action == 2:  # LEFT
            self.snake_dir = (-1, 0)
        elif action == 3:  # RIGHT
            self.snake_dir = (1, 0)

        if self.snake_dir != (0, 0):
            new_head = (self.snake_pos[-1][0] + self.snake_dir[0], self.snake_pos[-1][1] + self.snake_dir[1])
            self.snake_pos.append(new_head)

            if self.snake_pos[-1] == self.apple_pos:
                self.apple_pos = (random.randint(0, self.WIDTH - 1), random.randint(0, self.HEIGHT - 1))
                reward = 10  # Recompensa por comer a maçã
                self.score += reward
            else:
                self.snake_pos.pop(0)
                reward = -1  # Penalidade por movimento

            if (self.snake_pos[-1][0] < 0 or self.snake_pos[-1][0] >= self.WIDTH or
                self.snake_pos[-1][1] < 0 or self.snake_pos[-1][1] >= self.HEIGHT or
                self.snake_pos[-1] in self.snake_pos[:-1]):
                self.done = True
                reward = -10  # Grande penalidade por colisão

            return self.get_state(), reward, self.done

        return self.get_state(), reward, self.done
