import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque


# 定义多边形环境
class PolygonEnv:
    def __init__(self, num_nodes=5, custom_nodes=None):
        """
        初始化多边形环境。

        :param num_nodes: 如果未提供 custom_nodes，则生成正多边形的节点数
        :param custom_nodes: 自定义多边形的节点坐标，形状为 (num_nodes, 2)
        """
        if custom_nodes is not None:
            custom_nodes = np.array(custom_nodes)
            if custom_nodes.shape[0] < 3:
                raise ValueError("自定义多边形至少需要三个节点。")
            if custom_nodes.shape[1] != 2:
                raise ValueError("每个节点必须有 (x, y) 坐标。")
            self.original_nodes = custom_nodes
            self.num_original_nodes = len(custom_nodes)
        else:
            self.num_original_nodes = num_nodes
            self.original_nodes = self._generate_polygon_nodes(num_nodes)

        self.equidistant_nodes = self.generate_equidistant_nodes(interval=0.5)
        self.num_nodes = len(self.equidistant_nodes)
        self.nodes = self.equidistant_nodes
        self.current_node = 0  # 起点
        self.visited = set()
        self.visited.add(self.current_node)
        self.path = [self.current_node]

    def _generate_polygon_nodes(self, num_nodes):
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        nodes = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])  # 单位圆上的点
        return nodes

    def reset(self):
        self.current_node = 0
        self.visited = set()
        self.visited.add(self.current_node)
        self.path = [self.current_node]
        return self._get_state()

    def generate_equidistant_nodes(self, interval=0.5):
        """
        在多边形上生成等间隔的节点。

        :param interval: 节点之间的间隔距离
        :return: 等间隔节点的坐标列表
        """
        equidistant_points = []
        total_distance = 0.0
        num_original_nodes = self.num_original_nodes
        for i in range(num_original_nodes):
            start = self.original_nodes[i]
            end = self.original_nodes[(i + 1) % num_original_nodes]
            edge_length = np.linalg.norm(end - start)
            if edge_length == 0:
                continue  # 避免除以零
            num_steps = int(np.ceil(edge_length / interval))
            for j in range(num_steps + 1):
                t = j * interval / edge_length
                point = start + t * (end - start)
                equidistant_points.append(point)
        return np.array(equidistant_points)

    def step(self, action):
        next_node = action
        if next_node < 0 or next_node >= self.num_nodes:
            return self._get_state(), -1, True  # 非法动作

        if next_node in self.visited:
            return self._get_state(), -10, False  # 重复访问，给予负奖励

        current_pos = self.nodes[self.current_node]
        next_pos = self.nodes[next_node]
        distance = np.linalg.norm(next_pos - current_pos)
        reward = -distance

        # 鼓励对角线移动
        if next_node in self._get_non_adjacent_nodes(self.current_node):
            reward += 5  # 对角线移动的额外奖励

        self.current_node = next_node
        self.path.append(self.current_node)
        self.visited.add(self.current_node)

        done = len(self.visited) == self.num_nodes
        return self._get_state(), reward, done

    def _get_non_adjacent_nodes(self, node):
        """
        获取非相邻节点的索引。

        :param node: 当前节点索引
        :return: 非相邻节点索引列表
        """
        adjacent = [(node - 1) % self.num_nodes, (node + 1) % self.num_nodes]
        non_adjacent = [i for i in range(self.num_nodes) if i not in adjacent and i != node]
        return non_adjacent

    def _get_state(self):
        """
        获取当前状态，这里简化为当前节点的坐标。

        :return: 当前节点的坐标
        """
        return self.nodes[self.current_node].tolist()

    def render(self, show_equidistant=True):
        plt.figure(figsize=(8, 8))
        plt.title("任意多边形路径规划")
        original_nodes = self.original_nodes
        equidistant_nodes = self.nodes
        path = np.array([equidistant_nodes[i] for i in self.path])

        # 绘制原始多边形
        plt.plot(np.append(original_nodes[:, 0], original_nodes[0, 0]),
                 np.append(original_nodes[:, 1], original_nodes[0, 1]), 'r-', label='Original Polygon')

        # 绘制等间隔节点路径
        if show_equidistant:
            plt.plot(equidistant_nodes[:, 0], equidistant_nodes[:, 1], 'bo-', label='Equidistant Path')

        # 绘制实际访问路径
        plt.plot(path[:, 0], path[:, 1], 'go-', label='Visited Path')

        # 标记起点和终点
        plt.scatter(equidistant_nodes[0, 0], equidistant_nodes[0, 1], color='green', label='Start', s=100)
        plt.scatter(equidistant_nodes[-1, 0], equidistant_nodes[-1, 1], color='red', label='End', s=100)

        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
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

    def act(self, state, visited):
        if np.random.rand() <= self.epsilon:
            available_actions = [i for i in range(self.action_size) if i not in visited]
            if available_actions:
                return random.choice(available_actions)
            else:
                return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        act_values = act_values.squeeze(0).numpy()
        for i in visited:
            act_values[i] = -np.inf  # 将已访问节点的Q值设为负无穷
        return np.argmax(act_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * self.target_model(next_state).max(1)[0].item()
            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, target_f.clone())  # 错误的损失计算
            # 正确的损失计算
            target_tensor = torch.FloatTensor([target])
            loss = nn.MSELoss()(target_f, target_tensor.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# 训练过程
def train_dqn(env, agent, episodes=500, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, env.visited)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_model()
                print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


if __name__ == "__main__":
    # 定义一个自定义多边形
    custom_polygon = [
        [0, 0],
        [2, 0],
        [3, 1],
        [1, 3],
        [-1, 1]
    ]

    # 创建环境
    env = PolygonEnv(custom_nodes=custom_polygon)

    # 渲染环境及等间隔路径
    env.render(show_equidistant=True)

    # 初始化DQN代理，确保传递 state_size 和 action_size
    # 假设状态为节点坐标 (x, y)，因此 state_size=2
    # action_size 为等间隔节点的数量
    state_size = 2
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)  # 确保 DQNAgent 接受参数

    # 开始训练
    train_dqn(env, agent, episodes=500, batch_size=32)

    # 渲染最终路径
    env.render(show_equidistant=False)