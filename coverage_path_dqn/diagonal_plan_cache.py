import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# 初步思路是先人为构建出关键点
# 之后构建合适的约束条件用来学习即可

class PolygonEnv:
    def __init__(self, num_nodes=5, custom_nodes=None):
        """
        初始化多边形环境。
        """
        if custom_nodes is not None:
            self.nodes = np.array(custom_nodes)
        else:
            self.nodes = self._generate_polygon_nodes(num_nodes)
        self.polygon = self.nodes  # 简化为节点列表
        self.intersection_nodes = self.generate_intersection_nodes()
        self.num_nodes = len(self.intersection_nodes)
        self.path_nodes = np.array(self.intersection_nodes)
        self.current_node = 0
        self.visited = set([self.current_node])
        self.path = [self.current_node]

    def _generate_polygon_nodes(self, num_nodes):
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        nodes = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
        return nodes

    def generate_intersection_nodes(self):
        """
        生成由两条垂直直线构成的网格与多边形的交点作为路径节点。
        """
        intersections = set()
        center = np.mean(self.nodes, axis=0)
        num_lines = 5  # 每方向生成5条线
        min_x, min_y = np.min(self.nodes, axis=0)
        max_x, max_y = np.max(self.nodes, axis=0)

        # 生成水平线
        for i in range(num_lines):
            y = min_y + (max_y - min_y) * (i / (num_lines - 1))
            line = np.array([[min_x, y], [max_x, y]])  # 水平线的两个端点

            # 计算水平线与多边形边的交点
            for edge_idx in range(len(self.nodes)):
                edge_start = self.nodes[edge_idx]
                edge_end = self.nodes[(edge_idx + 1) % len(self.nodes)]
                inter = self.line_segment_intersection(line, [edge_start, edge_end])
                if inter is not None:
                    intersections.add(tuple(inter))

        # 生成垂直线
        for j in range(num_lines):
            x = min_x + (max_x - min_x) * (j / (num_lines - 1))
            line = np.array([[x, min_y], [x, max_y]])  # 垂直线的两个端点

            # 计算垂直线与多边形边的交点
            for edge_idx in range(len(self.nodes)):
                edge_start = self.nodes[edge_idx]
                edge_end = self.nodes[(edge_idx + 1) % len(self.nodes)]
                inter = self.line_segment_intersection(line, [edge_start, edge_end])
                if inter is not None:
                    intersections.add(tuple(inter))

        return np.array(list(intersections))

    @staticmethod
    def line_segment_intersection(line, edge):
        """
        计算直线与线段的交点。
        line 是 (N, 2) 的 NumPy 数组，表示直线
        edge 是 [start, end]，表示线段
        使用参数化方法计算交点
        """
        p1, p2 = line[0], line[1]  # 取直线的两个端点
        q1, q2 = edge[0], edge[1]  # 取线段的两个端点

        # 将线段表示为参数方程 p = p1 + t*(p2 - p1), q = q1 + u*(q2 - q1)
        r = p2 - p1
        s = q2 - q1

        # 手动计算二维叉积
        def cross2d(a, b):
            return a[0] * b[1] - a[1] * b[0]

        r_cross_s = cross2d(r, s)

        if abs(r_cross_s) < 1e-12:
            return None

        t_numerator = cross2d(q1 - p1, s)
        u_numerator = cross2d(q1 - p1, r)

        t = t_numerator / r_cross_s
        u = u_numerator / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p1 + t * r
            return intersection
        else:
            return None

    def reset(self):
        self.current_node = 0
        self.visited = set([self.current_node])
        self.path = [self.current_node]
        return self._get_state()

    def _get_state(self):
        return self.path_nodes[self.current_node]

    def step(self, action):
        next_node = action
        if next_node < 0 or next_node >= self.num_nodes:
            return self._get_state(), -1, True
        if next_node in self.visited:
            return self._get_state(), -10, False

        current_pos = self.path_nodes[self.current_node]
        next_pos = self.path_nodes[next_node]
        distance = np.linalg.norm(next_pos - current_pos)
        reward = -distance

        # 鼓励对角线移动
        if next_node in self._get_non_adjacent_nodes(self.current_node):
            reward += 5

        self.current_node = next_node
        self.visited.add(next_node)
        self.path.append(next_node)

        done = len(self.visited) == self.num_nodes
        return self._get_state(), reward, done

    def _get_non_adjacent_nodes(self, node):
        # 简化处理，实际应根据图结构确定
        return [i for i in range(self.num_nodes) if i != node]

    def render(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.nodes[:, 0], self.nodes[:, 1], 'r-', label='Polygon')
        plt.plot(self.path_nodes[:, 0], self.path_nodes[:, 1], 'bo-', label='Path Nodes')
        plt.scatter(self.path_nodes[0, 0], self.path_nodes[0, 1], color='green', label='Start', s=100)
        plt.scatter(self.path_nodes[-1, 0], self.path_nodes[-1, 1], color='red', label='End', s=100)
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
                print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
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
    env.render()

    # 初始化DQN代理，确保传递 state_size 和 action_size
    state_size = 2  # 使用节点坐标 (x, y)
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)

    # 开始训练
    train_dqn(env, agent, episodes=500, batch_size=32)

    # 渲染最终路径
    env.render()