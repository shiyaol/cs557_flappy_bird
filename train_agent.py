import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
from src.deep_q_shiyao import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing
import cv2
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

image_size = 84
batch_size = 32
learning_rate = 1e-6
gamma = 0.99
iter_num = 200
memo_size = 50000
log_path = "tensorboard"
model_path = "trained_models"

def pre_processing(image, width, height):
    gray_image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, bin_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    #cv2.imshow("image", image)
    return bin_image[None, :, :].astype(np.float32)

def train_agent(iter_num = iter_num, im_size = image_size, b_size = batch_size, lr = learning_rate, gamma = gamma, m_size = memo_size, m_path = model_path):

    torch.manual_seed(123)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], im_size, im_size)
    image = torch.from_numpy(image)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    replay_memory = []
    iter = 0

    while iter < iter_num:
        prediction = model(state)[0]
        #print(image)
        # Exploration or exploitation
        epsilon = 0.0001 + ((iter_num - iter) * (0.1 - 0.0001) / iter_num)
        random_prob = random()

        if random_prob <= epsilon:
            print("random")
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction)
            
        next_frame_image, reward, terminal = game_state.next_frame(action)
        next_frame_image = torch.from_numpy(pre_processing(next_frame_image[:game_state.screen_width, :int(game_state.base_y)], im_size, im_size))

        train_image = torch.cat((state[0, 1:, :, :], next_frame_image))[None, :, :, :]

        if len(replay_memory) < m_size:
            replay_memory.append([state, action, reward, train_image, terminal])
        else:
            del replay_memory[0]
            replay_memory.append([state, action, reward, train_image, terminal])

        train_batch = sample(replay_memory, min(len(replay_memory), b_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*train_batch)

        train_state = torch.cat(tuple(s for s in state_batch))
        actions = torch.from_numpy(np.array([[1, 0] if a == 0 else [0, 1] for a in action_batch], dtype=np.float32))
        rewards = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        train_next = torch.cat(tuple(ns for ns in next_state_batch))

        current_prediction_batch = model(train_state)
        next_prediction_batch = model(train_next)

        y_batch = torch.cat(tuple(r if t else r + gamma * torch.max(next_p) for r, t, next_p in zip(rewards, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * actions, dim=1)

        optimizer.zero_grad()

        loss = loss_function(q_value, y_batch)
        loss.backward()
        optimizer.step()
        state = train_image
        iter += 1

        print("Iteration: {}/{}, Action: {}, Loss: {}, Reward: {}, Q-value: {}".format(iter, iter_num, action, loss, reward, torch.max(prediction)))

        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(m_path, iter+1))

    torch.save(model, "{}/flappy_bird_final".format(m_path))

train_agent()