import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from collections import deque


# 定义环境
class PolygonEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.agent_pos = [0, 0]
        self.visited = set()
        self.visited.add(tuple(self.agent_pos))
        self.path = [self.agent_pos]  # 初始化路径

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos = [0, 0]
        self.visited = set()
        self.visited.add(tuple(self.agent_pos))
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:   # 上
            x -= 1
        elif action == 1: # 下
            x += 1
        elif action == 2: # 左
            y -= 1
        elif action == 3: # 右
            y += 1
        elif action == 4: # 左上
            x -= 1
            y -= 1
        elif action == 5: # 右上
            x -= 1
            y += 1
        elif action == 6: # 左下
            x += 1
            y -= 1
        elif action == 7: # 右下
            x += 1
            y += 1

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.agent_pos = [x, y]
            self.path.append(self.agent_pos)  # 更新路径
            if tuple(self.agent_pos) not in self.visited:
                self.visited.add(tuple(self.agent_pos))
                reward = 1
            else:
                reward = -1
            done = len(self.visited) == self.grid_size * self.grid_size
            return self.agent_pos, reward, done
        else:
            return self.agent_pos, -1, True

    def render(self):
        plt.figure(figsize=(5, 5))
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.grid()

        # 绘制覆盖区域
        for (x, y) in self.visited:
            plt.fill([x, x + 1, x + 1, x], [y, y, y + 1, y + 1], color='lightblue', alpha=0.5)

        # 绘制路径
        path_x, path_y = zip(*self.path)
        plt.plot(path_x, path_y, marker='o', color='b', label='Path')

        # 绘制起点和终点
        plt.scatter(path_x[0], path_y[0], color='green', label='Start', s=100)
        plt.scatter(path_x[-1], path_y[-1], color='red', label='End', s=100)

        plt.legend()
        plt.title("Coverage Path Planning")
        plt.show()

# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # 确保形状为 [1, state_size]
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f = target_f.squeeze(0)  # 将形状从 [1, action_size] 转换为 [action_size]
            target_f[action] = target  # 直接索引
            target_f = target_f.unsqueeze(0)  # 恢复形状为 [1, action_size]
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 训练过程
env = PolygonEnv(grid_size=10)
state_size = 2
action_size = 8
agent = DQNAgent(state_size, action_size)
batch_size = 32
episodes = 700

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            agent.update_target_model()
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

env.render()