import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# 多边形环境类
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
        self.current_direction = None  # 当前移动方向
        self.direction_weight = 5.0  # 方向一致的奖励权重
        self.adjacent_weight = 3.0  # 相邻节点的奖励权重
        self.non_adjacent_penalty = -2.0  # 非相邻节点的惩罚

    def _generate_polygon_nodes(self, num_nodes):
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        nodes = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
        return nodes

    def load_intersections_from_file(self, file_path):
        try:
            points = np.loadtxt(file_path, delimiter=',')
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("文件中的点必须具有二维坐标 (x, y)。")
            intersections = set(tuple(point) for point in points)
            return intersections
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。请检查文件路径。")
            return []
        except ValueError as ve:
            print(f"数据格式错误: {ve}")
            return []

    def generate_intersection_nodes(self):
        file_path = "/home/aiforce/0307_deep_learning/learn_base_plan/coverage_path_dqn/output_points.txt"
        intersections = self.load_intersections_from_file(file_path)
        if len(intersections) == 0:
            print("未加载到任何交点，可能需要检查文件内容或路径。")
        return np.array(list(intersections))

    def reset(self):
        self.current_node = 0
        self.visited = set([self.current_node])
        self.path = [self.current_node]
        self.current_direction = None  # 重置方向
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
        reward = -distance  # 基础奖励为负距离

        # 计算移动方向
        move_vector = next_pos - current_pos
        move_direction = np.arctan2(move_vector[1], move_vector[0])

        # 如果当前方向为空，设置为移动方向
        if self.current_direction is None:
            self.current_direction = move_direction
        else:
            # 计算方向一致性
            direction_diff = abs(move_direction - self.current_direction)
            if direction_diff < np.pi / 4:  # 方向一致
                reward += self.direction_weight

        # 检查是否移动到相邻节点
        if next_node in self._get_adjacent_nodes(self.current_node):
            reward += self.adjacent_weight
        else:
            reward += self.non_adjacent_penalty  # 非相邻节点惩罚

        # 更新当前节点和方向
        self.current_node = next_node
        self.visited.add(next_node)
        self.path.append(next_node)
        self.current_direction = move_direction

        done = len(self.visited) == self.num_nodes
        return self._get_state(), reward, done

    def _get_adjacent_nodes(self, node):
        # 简化处理，假设相邻节点是索引相邻的节点
        adjacent_nodes = []
        if node > 0:
            adjacent_nodes.append(node - 1)
        if node < self.num_nodes - 1:
            adjacent_nodes.append(node + 1)
        return adjacent_nodes

    def _get_non_adjacent_nodes(self, node):
        # 简化处理，实际应根据图结构确定
        return [i for i in range(self.num_nodes) if i != node]

    def render(self):
        plt.ioff()  # 禁用交互模式
        plt.figure(figsize=(8, 8))
        plt.plot(self.nodes[:, 0], self.nodes[:, 1], 'r-', label='Polygon')
        plt.plot(self.path_nodes[:, 0], self.path_nodes[:, 1], 'bo-', label='Path Nodes')
        plt.scatter(self.path_nodes[0, 0], self.path_nodes[0, 1], color='green', label='Start', s=100)
        plt.scatter(self.path_nodes[-1, 0], self.path_nodes[-1, 1], color='red', label='End', s=100)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=True)  # 阻塞程序，直到用户关闭窗口

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
            loss = nn.MSELoss()(target_f, torch.FloatTensor([target]).unsqueeze(0))
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
    custom_polygon = []

    with open("polygonpts.txt", 'r') as file:
        for line in file:
            point = line.strip().split()
            x = float(point[0])
            y = float(point[1])
            custom_polygon.append([x, y])

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