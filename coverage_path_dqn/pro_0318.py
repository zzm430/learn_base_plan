import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point,MultiPoint,MultiLineString,GeometryCollection
from collections import deque

# 初步思路是先人为构建出关键点
# 之后构建合适的约束条件用来学习即可
class PolygonEnv:
    def __init__(self, num_nodes=5, swath= None, custom_nodes=None):
        """
        初始化多边形环境。
        """
        if custom_nodes is not None:
            self.nodes = np.array(custom_nodes)
        else:
            self.nodes = self._generate_polygon_nodes(num_nodes)
        self.first_direction = None
        self.second_direction = None
        self.polygon = self.nodes  # 简化为节点列表
        self.swath = swath
        self.intersection_nodes = self.generate_intersection_nodes()
        self.current_node = 0
        self.visited = set([self.current_node])
        self.path = [self.current_node]
        self.num_nodes = len(self.intersection_nodes)
        self.path_nodes = np.array(self.intersection_nodes)
        self.current_direction = 'first'  # 当前方向，初始为第一方向
    def _generate_polygon_nodes(self, num_nodes):
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        nodes = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
        return nodes


    def load_intersections_from_file(self, file_path):
        try:
            # 使用numpy的loadtxt读取文件，假设文件中每行是两个浮点数，用空格或逗号分隔
            # 如果使用逗号分隔，可以设置delimiter=','
            points = np.loadtxt(file_path, delimiter=',')

            # 确保读取到的点具有二维坐标
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("文件中的点必须具有二维坐标 (x, y)。")

            # 将读取到的点转换为列表并存储到intersections
            intersections = set(tuple(point) for point in points)

            return intersections

        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。请检查文件路径。")
            return []
        except ValueError as ve:
            print(f"数据格式错误: {ve}")
            return []

    def read_polygon_from_file(self,file_path):
        polygon = []
        with open(file_path,'r') as file:
            for line in file:
                x,y = map(float,line.strip().split())
                polygon.append((x,y)) #将点存为元组
        return polygon

    def calculate_distance(self,point_1,point_2):
        #计算两点之间的欧几里得距离**
        return math.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

    def find_longest_edge(self,polygon):
        longest_length = 0
        longest_edge = None

        for i in range(len(polygon)):
            point1 = polygon[i]
            point2 = polygon[(i + 1) % len(polygon)]

            #计算边的长度
            length = self.calculate_distance(point1,point2)

            #更新最长边
            if length > longest_length:
                longest_length = length
                longest_edge = (point1,point2)
        return longest_edge,longest_length

    def distance_point_to_line_with_foot(self, P, A, B):
        """
        计算点 P 到直线 AB 的距离，并返回垂足点。
        :param P: 点的坐标 (x, y)
        :param A: 直线起点 (x1, y1)
        :param B: 直线终点 (x2, y2)
        :return: 距离和垂足点坐标
        """
        x, y = P
        x1, y1 = A
        x2, y2 = B

        # 计算直线方程系数 A, B, C
        A_coeff = y2 - y1
        B_coeff = x1 - x2
        C_coeff = x2 * y1 - x1 * y2

        # 计算点到直线的距离
        numerator = abs(A_coeff * x + B_coeff * y + C_coeff)
        denominator = math.sqrt(A_coeff ** 2 + B_coeff ** 2)
        distance = numerator / denominator

        # 计算垂足点
        dx = x2 - x1
        dy = y2 - y1
        segment_length_squared = dx ** 2 + dy ** 2

        if segment_length_squared == 0:
            # 如果线段长度为 0，垂足点为 A
            foot_x, foot_y = x1, y1
        else:
            # 计算投影参数 t
            t = ((x - x1) * dx + (y - y1) * dy) / segment_length_squared
            foot_x = x1 + t * dx
            foot_y = y1 + t * dy

        return distance, (foot_x, foot_y)

    def compute_max_distance_farest(self,polygon,longest_edge):
        longest_distance = 0
        foot_point = None    #垂足点
        farthest_point = None

        for i in range(len(polygon)):
            point1 = polygon[i]
            dis ,foot = self.distance_point_to_line_with_foot(point1,longest_edge[0],longest_edge[1])
            if dis > longest_distance:
               longest_distance = dis
               farthest_point = point1
               foot_point = foot

        print("The max distance is:", longest_distance)
        print("The farthest point is:", farthest_point)
        print("The foot point is:", foot_point)
        return longest_distance, farthest_point, foot_point

    def translate_edge(self, edge, direction, distance):
        """
        将线段沿着指定方向平移固定距离，并在两端延长100米。
        :param edge: 线段的两个端点，格式为 [(x1, y1), (x2, y2)]
        :param direction: 平移方向，格式为 (dx, dy)，表示方向向量
        :param distance: 平移距离
        :return: 平移并延长后的线段，格式为 [(new_x1, new_y1), (new_x2, new_y2)]
        """
        # 计算方向向量的单位向量
        dx, dy = direction
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        if magnitude == 0:
            raise ValueError("Direction vector cannot be zero.")

        unit_dx = dx / magnitude
        unit_dy = dy / magnitude

        # 计算平移向量
        translate_dx = unit_dx * distance
        translate_dy = unit_dy * distance

        # 对线段的两个端点进行平移
        new_point1 = (edge[0][0] + translate_dx, edge[0][1] + translate_dy)
        new_point2 = (edge[1][0] + translate_dx, edge[1][1] + translate_dy)

        # 计算线段的方向向量
        edge_dx = new_point2[0] - new_point1[0]
        edge_dy = new_point2[1] - new_point1[1]
        edge_magnitude = math.sqrt(edge_dx ** 2 + edge_dy ** 2)
        if edge_magnitude == 0:
            raise ValueError("Edge length cannot be zero.")

        unit_edge_dx = edge_dx / edge_magnitude
        unit_edge_dy = edge_dy / edge_magnitude

        # 延长线段两端各100米
        extended_point1 = (new_point1[0] - unit_edge_dx * 100, new_point1[1] - unit_edge_dy * 100)
        extended_point2 = (new_point2[0] + unit_edge_dx * 100, new_point2[1] + unit_edge_dy * 100)

        return [extended_point1, extended_point2]

    def extend_line_segment(self,line, extension_length=100):
        """
        将线段在其两端各延长指定长度。
        :param line: 线段的两个端点，格式为 [(x1, y1), (x2, y2)]
        :param extension_length: 延长的长度（单位：米）
        :return: 延长后的线段，格式为 [(new_x1, new_y1), (new_x2, new_y2)]
        """
        # 提取起点和终点
        (x1, y1), (x2, y2) = line

        # 计算方向向量
        dx = x2 - x1
        dy = y2 - y1

        # 计算方向向量的长度
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            raise ValueError("线段的长度为 0，无法计算方向向量。")

        # 归一化方向向量
        unit_dx = dx / length
        unit_dy = dy / length

        # 延长起点
        new_start = (x1 - unit_dx * extension_length, y1 - unit_dy * extension_length)

        # 延长终点
        new_end = (x2 + unit_dx * extension_length, y2 + unit_dy * extension_length)

        return [new_start, new_end]

    def generate_intersection_nodes(self):
        #从某文件中读取polygon信息
        file_path = "/home/aiforce/0307_deep_learning/learn_base_plan/coverage_path_dqn/polygonpts.txt"
        polygon = self.read_polygon_from_file(file_path)
        all_origin_line = []
        all_intersection_nodes = []
        print("读取的多边形坐标点：")
        for point in polygon:
            print(point)
        #求取多边形最长的一条边
        longest_edge,longest_length = self.find_longest_edge(polygon)
        max_distance ,farthest_point,foot_point = self.compute_max_distance_farest(polygon,longest_edge)
        max_number = max_distance // self.swath
        print("the max_number is :",max_number)
        direction = (farthest_point[0] - foot_point[0] ,farthest_point[1] - foot_point[1])
        self.first_direction = direction
        trans_polygon = Polygon(polygon)
        #第一方向
        for i in range(1,int(max_number)):
            move_dis = i * self.swath
            ordered_move_line = self.translate_edge(longest_edge,direction,move_dis)
            all_origin_line.append(ordered_move_line)
        #第二方向
        second_direction = (longest_edge[0][0] - longest_edge[1][0],longest_edge[0][1] - longest_edge[1][1])
        self.second_direction = second_direction
        a_second_direction = (longest_edge[0][1] - longest_edge[1][0],longest_edge[1][1] - longest_edge[0][1])
        line = []
        line.append(foot_point)
        line.append(farthest_point)
        second_line = self.extend_line_segment(line)
        for i in range(0,int(50)):
            move_dis = i * self.swath
            ordered_move_line = self.translate_edge(line,second_direction,move_dis)
            all_origin_line.append(ordered_move_line)
        for i in range(0, int(50)):
            move_dis = i * self.swath
            ordered_move_line_two = self.translate_edge(line, a_second_direction, move_dis)
            all_origin_line.append(ordered_move_line_two)

        for i in all_origin_line:
            trans_line = LineString(i)
            intersection = trans_polygon.intersection(trans_line)
            if intersection.is_empty:
                continue  # 如果没有交点，跳过
            elif isinstance(intersection, Point):
                # 如果是点，直接存储坐标
                all_intersection_nodes.append(intersection.coords[0])
            elif isinstance(intersection, MultiPoint):
                # 如果是多点，拆分为多个点并存储
                for point in intersection.geoms:
                    all_intersection_nodes.append(point.coords[0])
            elif isinstance(intersection, LineString):
                # 如果是线段，存储起点和终点
                all_intersection_nodes.append(intersection.coords[0])
                all_intersection_nodes.append(intersection.coords[-1])
            elif isinstance(intersection, MultiLineString):
                # 如果是多线段，遍历所有线段并存储起点和终点
                for line in intersection.geoms:
                    all_intersection_nodes.append(line.coords[0])
                    all_intersection_nodes.append(line.coords[-1])
            elif isinstance(intersection, GeometryCollection):
                # 如果是几何集合，遍历所有几何对象并处理
                for geom in intersection.geoms:
                    if isinstance(geom, Point):
                        all_intersection_nodes.append(geom.coords[0])
                    elif isinstance(geom, LineString):
                        all_intersection_nodes.append(geom.coords[0])
                        all_intersection_nodes.append(geom.coords[-1])
                    elif isinstance(geom, MultiPoint):
                        for point in geom.geoms:
                            all_intersection_nodes.append(point.coords[0])
                    elif isinstance(geom, MultiLineString):
                        for line in geom.geoms:
                            all_intersection_nodes.append(line.coords[0])
                            all_intersection_nodes.append(line.coords[-1])
            else:
                # 处理其他未知类型（如 Polygon）
                raise ValueError(f"未知的几何类型: {type(intersection)}")
        return all_intersection_nodes

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

        # 计算移动方向
        move_direction = next_pos - current_pos
        move_direction_norm = np.linalg.norm(move_direction)
        if move_direction_norm == 0:
            unit_move_direction = np.array([0, 0])
        else:
            unit_move_direction = move_direction / move_direction_norm

        # 根据当前方向给予奖励
        if self.current_direction == 'first':
            if self.first_direction is not None:
                first_direction_norm = np.linalg.norm(self.first_direction)
                if first_direction_norm > 0:
                    unit_first_direction = self.first_direction / first_direction_norm
                    dot_product_first = np.dot(unit_move_direction, unit_first_direction)
                    reward += 100 * dot_product_first  # 奖励与第一方向的一致性
        else:
            if self.second_direction is not None:
                second_direction_norm = np.linalg.norm(self.second_direction)
                if second_direction_norm > 0:
                    unit_second_direction = self.second_direction / second_direction_norm
                    dot_product_second = np.dot(unit_move_direction, unit_second_direction)
                    reward += 100 * dot_product_second  # 奖励与第二方向的一致性

        # 更新当前节点和路径
        self.current_node = next_node
        self.visited.add(next_node)
        self.path.append(next_node)

        # 检查是否完成
        done = len(self.visited) == self.num_nodes

        # 如果没有完成，自动选择下一个节点
        if not done:
            if self.current_direction == 'first':
                # 沿着第一方向走到下一个节点
                next_node = self._find_next_node_in_direction(self.first_direction)
                # 切换到第二方向，找到最近的起始节点
                start_node = self._find_nearest_node_in_direction(self.second_direction, next_node)
                self.current_direction = 'second'  # 切换方向
                # 移动到第二方向的起始节点
                return self.step(start_node)
            else:
                # 沿着第二方向走到下一个节点
                next_node = self._find_next_node_in_direction(self.second_direction)
                # 切换到第一方向，找到最近的起始节点
                start_node = self._find_nearest_node_in_direction(self.first_direction, next_node)
                self.current_direction = 'first'  # 切换方向
                # 移动到第一方向的起始节点
                return self.step(start_node)
        else:
            return self._get_state(), reward, done

    def _find_next_node_in_direction(self, direction):
        """
        沿着指定方向找到下一个未访问节点。
        :param direction: 方向向量
        :return: 下一个未访问节点的索引
        """
        current_pos = self.path_nodes[self.current_node]
        nearest_node = None
        min_distance = float('inf')

        for i in range(self.num_nodes):
            if i not in self.visited:
                node_pos = self.path_nodes[i]
                move_direction = node_pos - current_pos
                move_direction_norm = np.linalg.norm(move_direction)
                if move_direction_norm == 0:
                    unit_move_direction = np.array([0, 0])
                else:
                    unit_move_direction = move_direction / move_direction_norm

                # 计算与目标方向的一致性
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    unit_direction = direction / direction_norm
                    dot_product = np.dot(unit_move_direction, unit_direction)
                    if dot_product > 0.9:  # 方向一致性阈值
                        distance = np.linalg.norm(node_pos - current_pos)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_node = i

        return nearest_node if nearest_node is not None else self.current_node

    def _find_nearest_node_in_direction(self, direction, start_node):
        """
        沿着指定方向找到最近的未访问节点，作为下一个方向的起始节点。
        :param direction: 方向向量
        :param start_node: 起始节点索引
        :return: 最近的未访问节点的索引
        """
        start_pos = self.path_nodes[start_node]
        nearest_node = None
        min_distance = float('inf')

        for i in range(self.num_nodes):
            if i not in self.visited:
                node_pos = self.path_nodes[i]
                move_direction = node_pos - start_pos
                move_direction_norm = np.linalg.norm(move_direction)
                if move_direction_norm == 0:
                    unit_move_direction = np.array([0, 0])
                else:
                    unit_move_direction = move_direction / move_direction_norm

                # 计算与目标方向的一致性
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    unit_direction = direction / direction_norm
                    dot_product = np.dot(unit_move_direction, unit_direction)
                    if dot_product > 0.9:  # 方向一致性阈值
                        distance = np.linalg.norm(node_pos - start_pos)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_node = i

        return nearest_node if nearest_node is not None else start_node

    def _get_non_adjacent_nodes(self, node):
        # 简化处理，实际应根据图结构确定
        return [i for i in range(self.num_nodes) if i != node]

    def render(self):
        plt.ioff()  #禁用交互模式
        plt.figure(figsize=(8, 8))
        plt.plot(self.nodes[:, 0], self.nodes[:, 1], 'r-', label='Polygon')
        plt.plot(self.path_nodes[:, 0], self.path_nodes[:, 1], 'bo-', label='Path Nodes')
        plt.scatter(self.path_nodes[0, 0], self.path_nodes[0, 1], color='green', label='Start', s=100)
        plt.scatter(self.path_nodes[-1, 0], self.path_nodes[-1, 1], color='red', label='End', s=100)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=True)  #阻塞程序，直到用户关闭窗口

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
    custom_polygon = []

    with open("polygonpts.txt",'r') as file:
        for line in file:
            point = line.strip().split()

            x = float(point[0])
            y = float(point[1])

            custom_polygon.append([x,y])

    # 创建环境
    env = PolygonEnv(swath= 8,custom_nodes=custom_polygon)

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