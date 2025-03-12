import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# 定义多边形环境
class PolygonEnv:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = self._generate_polygon_nodes(num_nodes)  # 生成多边形节点
        self.current_node = 0  # 起点
        self.visited = set()  # 已访问的节点
        self.visited.add(self.current_node)
        self.path = [self.current_node]  # 路径

    def _generate_polygon_nodes(self, num_nodes):
        # 生成一个正多边形的节点坐标
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        nodes = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])  # 单位圆上的点
        return nodes

    def reset(self):
        self.current_node = 0
        self.visited = set()
        self.visited.add(self.current_node)
        self.path = [self.current_node]
        return self.current_node

    def step(self, action):
        next_node = action
        if next_node < 0 or next_node >= self.num_nodes:
            return self.current_node, -1, True  # 非法动作

        # 计算奖励：鼓励对角线路径
        current_pos = self.nodes[self.current_node]
        next_pos = self.nodes[next_node]
        distance = np.linalg.norm(next_pos - current_pos)  # 计算两点之间的欧几里得距离
        reward = -distance  # 距离越小，奖励越大（鼓励直线路径）

        self.current_node = next_node
        self.path.append(self.current_node)
        self.visited.add(self.current_node)

        done = len(self.visited) == self.num_nodes  # 是否访问完所有节点
        return self.current_node, reward, done

    def render(self):
        plt.figure(figsize=(5, 5))
        plt.title("Polygon Path Planning")
        nodes = self.nodes
        path = np.array([nodes[i] for i in self.path])

        # 绘制多边形
        plt.plot(np.append(nodes[:, 0], nodes[0, 0]), np.append(nodes[:, 1], nodes[0, 1]), 'k-', label='Polygon')

        # 绘制路径
        plt.plot(path[:, 0], path[:, 1], 'bo-', label='Path')

        # 标记起点和终点
        plt.scatter(nodes[0, 0], nodes[0, 1], color='green', label='Start', s=100)
        plt.scatter(nodes[-1, 0], nodes[-1, 1], color='red', label='End', s=100)

        plt.legend()
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
        state = torch.FloatTensor(state).unsqueeze(0)
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
            target_f = target_f.squeeze(0)
            target_f[action] = target
            target_f = target_f.unsqueeze(0)
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 训练过程
num_nodes = 10  # 多边形的节点数
env = PolygonEnv(num_nodes)
state_size = 2  # 每个节点的坐标 (x, y)
action_size = num_nodes  # 动作空间为节点数
agent = DQNAgent(state_size, action_size)
batch_size = 32
episodes = 500

for e in range(episodes):
    state = env.reset()
    state = env.nodes[state]  # 获取当前节点的坐标
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = env.nodes[next_state]  # 获取下一个节点的坐标
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