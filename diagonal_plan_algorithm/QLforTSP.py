import random
import math
import numpy as np
from itertools import combinations
import copy
import matplotlib.pyplot as plt
from citynode import CityNode
from learn_base_plan.diagonal_plan_algorithm.citynode import NodeType
from shapely.geometry import Polygon, LineString, Point,MultiPoint,MultiLineString,GeometryCollection
from matplotlib.animation import FuncAnimation

class tsp:
    def __init__(self,node_num):
        self.swath = 7
        self.extra_cost = 5
        self.dummy_cost = 0
        self.node_pos,self.all_intersection_nodes_vector = self.generate_node_pos()  # 随机生成node_num个站点
        self.node_num = len(self.node_pos)
        self.node_pos_dict = {cnt:pos for cnt,pos in zip(range(len(self.node_pos)), self.node_pos)}
        self.dist_dict = self.cal_dist(node_pos=self.node_pos)
        self.stops = []  # 按顺序记录依次经过了哪些stop
        self.city_nodes = self.generate_city_nodes()
        self.dis_martix = self.reward_matrix()
        self.origin_render()

    def origin_render(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Show stops
        # end_x = self.node_pos[self.stops[0]][0]
        # end_y = self.node_pos[self.stops[0]][1]
        # self.x = np.concatenate([np.array([x[0] for x in self.node_pos]) , np.array([end_x])],axis=0)
        # self.y = np.concatenate([np.array([x[1] for x in self.node_pos]) , np.array([end_y])],axis=0)
        x_coords = np.array([x[0] for x in self.node_pos])
        y_coords = np.array([x[1] for x in self.node_pos])
        # ax.scatter(self.x, self.y, c="red", s=50)
        plt.scatter(x_coords,y_coords,c='red',s=50)

        for segment in self.all_intersection_nodes_vector:
            if len(segment) == 2:
                x_segment = [segment[0][0],segment[1][0]]
                y_segment = [segment[0][1],segment[1][1]]
                plt.plot(x_segment,y_segment,'b-',lw=2,alpha= 0.7)

        plt.title("tsp nodes visualization")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_node_sum(self):
        return self.node_num

    def generate_city_nodes(self):
        real_track_id = 0
        city_temp_nodes = []
        for i in self.all_intersection_nodes_vector:
            x_diff = i[0][0] - i[1][0]
            y_diff = i[0][1] - i[1][1]
            direction = (x_diff,y_diff)
            angle_direction = math.atan2(direction[1], direction[0]) * 180 / math.pi
            real_track_id += 1
            distance = ((i[0][0] - i[1][0]) ** 2 + (i[0][1] - i[1][1]) ** 2) ** 0.5
            node_A = CityNode(i[0], angle_direction, NodeType.ENDPOINT, i[1], real_track_id, distance)
            node_B = CityNode(i[1], angle_direction, NodeType.ENDPOINT, i[0], real_track_id, distance)
            city_temp_nodes.append(node_A)
            city_temp_nodes.append(node_B)
        return  city_temp_nodes

    @staticmethod
    def find_intersection(line1:LineString,line2:LineString):
        if line1.intersection(line2):
            intersection = line1.intersection(line2)
            if intersection.geom_type == 'Point':
                return { "intersect_flag":True,"point":(intersection.x,intersection.y)}
        else:
            return {"intersect_flag":False,"point":(0,0)}

    @staticmethod
    def compute_angle(v1, v2):
        """
        计算从向量v1到向量v2的角度，逆时针为正，范围是-MPI到MPI
        v1:第一个二维向量
        v2:第二个二维向量
        """
        v1 = np.asarray(v1).flatten()
        v2 = np.asarray(v2).flatten()

        if v1.shape != (2,) or v2.shape != (2,):
            raise ValueError("输入向量必须是二维的")

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            raise ValueError("输入向量不能是零向量")

        v1_norm = v1 / norm_v1
        v2_norm = v2 / norm_v2

        dot = np.dot(v1_norm, v2_norm)
        det = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]

        theta = np.arctan2(det, dot)
        return theta

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

    def read_polygon_from_file(self,file_path):
        polygon = []
        with open(file_path,'r') as file:
            for line in file:
                x,y = map(float,line.strip().split())
                polygon.append((x,y)) #将点存为元组
        return polygon

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

    def generate_node_pos(self):  # 生成浮点坐标
        file_path = "/home/aiforce/0307_deep_learning/learn_base_plan/coverage_path_dqn/polygonpts2.txt"
        polygon = self.read_polygon_from_file(file_path)
        all_origin_line = []
        all_intersection_nodes = []
        all_intersection_nodes_vector = []
        print("读取的多边形坐标点：")
        for point in polygon:
            print(point)
        # 求取多边形最长的一条边
        longest_edge, longest_length = self.find_longest_edge(polygon)
        max_distance, farthest_point, foot_point = self.compute_max_distance_farest(polygon, longest_edge)
        max_number = max_distance // self.swath
        print("the max_number is :", max_number)
        direction = (farthest_point[0] - foot_point[0], farthest_point[1] - foot_point[1])
        self.first_direction = direction
        trans_polygon = Polygon(polygon)
        # 第一方向
        for i in range(1, int(max_number + 2)):
            move_dis = i * self.swath
            ordered_move_line = self.translate_edge(longest_edge, direction, move_dis)
            all_origin_line.append(ordered_move_line)
        # 第二方向
        second_direction = (longest_edge[0][0] - longest_edge[1][0], longest_edge[0][1] - longest_edge[1][1])
        self.second_direction = second_direction
        a_second_direction = (longest_edge[0][1] - longest_edge[1][0], longest_edge[1][1] - longest_edge[0][1])
        line = []
        line.append(foot_point)
        line.append(farthest_point)
        second_line = self.extend_line_segment(line)
        # 先将second_line延长到能够完整覆盖整个多边形轮廓
        use_second_line = self.translate_edge(second_line, a_second_direction, 300)

        for i in range(0, int(1000)):
            move_dis = i * self.swath
            ordered_move_line = self.translate_edge(use_second_line, second_direction, move_dis)
            all_origin_line.append(ordered_move_line)
        # for i in range(0, int(50)):
        #     move_dis = i * self.swath
        #     ordered_move_line_two = self.translate_edge(line, a_second_direction, move_dis)
        #     all_origin_line.append(ordered_move_line_two)

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
                list_point = [intersection.coords[0], intersection.coords[-1]]
                all_intersection_nodes_vector.append(list_point)
            elif isinstance(intersection, MultiLineString):
                # 如果是多线段，遍历所有线段并存储起点和终点
                for line in intersection.geoms:
                    all_intersection_nodes.append(line.coords[0])
                    all_intersection_nodes.append(line.coords[-1])
                    list_point = [intersection.coords[0], intersection.coords[-1]]
                    all_intersection_nodes_vector.append(list_point)
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
        return all_intersection_nodes,all_intersection_nodes_vector

    def cal_dist(self,node_pos):
        distances_dict = {}
        for pair in combinations(node_pos, 2):
            distance = self.calculate_distance(pair[0], pair[1])
            distances_dict[pair] = distance
            distances_dict[(pair[1],pair[0])] = distance  # 对称
        return distances_dict

    def calculate_distance(self, point1, point2):
        # 计算两点之间的欧几里得距离
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def reward_matrix(self):
        # rew_mat = np.zeros([node_num,node_num+1])  # 最后一列放到end的reward，end和start是一个点
        # for cnt_x in range(node_num):  # row
        #     for cnt_y in range(node_num):  # col  # 可考虑对称，减少计算量
        #         if cnt_x==cnt_y:  # 下一步不能返回自己
        #             rew_mat[cnt_x][cnt_y] = np.NAN
        #         else:
        #             rew_mat[cnt_x][cnt_y] = -self.dist_dict.get((self.node_pos[cnt_x], self.node_pos[cnt_y]))
                    # rew_mat[cnt_x][cnt_y] = self.end_rew -1*self.dist_dict.get((self.node_pos_dict.get(cnt_x),self.node_pos_dict.get(cnt_y)))
        # for cnt in range(node_num):
        #     # if cnt==self.stops[0]:
        #     #     rew_mat[cnt][node_num] = np.NAN
        #     # else:
        #     rew_mat[cnt][node_num] = self.end_rew + rew_mat[cnt][0]
        size_citys = len(self.city_nodes)
        dis_martix = np.zeros((size_citys, size_citys), dtype=float)
        for i in range(size_citys):
            for j in range(i + 1):
                if i == j:
                    dis_martix[i][j] = np.NAN
                else:
                    if self.city_nodes[i].node_type == NodeType.ENDPOINT and self.city_nodes[
                        j].node_type == NodeType.ENDPOINT:
                        if self.city_nodes[i].real_track_id == self.city_nodes[j].real_track_id:
                            dis_martix[i][j] = 0
                        else:
                            cost = ((self.city_nodes[i].position[0] - self.city_nodes[j].position[0]) ** 2 + (
                                        self.city_nodes[i].position[1] - self.city_nodes[j].position[1] ) ** 2) ** 0.5
                            if self.city_nodes[i].direction == self.city_nodes[j].direction:
                                cost += self.extra_cost
                                if abs(self.city_nodes[i].real_track_id - self.city_nodes[j].real_track_id) == 1:
                                    cost += self.extra_cost
                                if (self.city_nodes[i].current_track_length + self.city_nodes[
                                    j].current_track_length < 100 and
                                        abs(self.city_nodes[i].real_track_id - self.city_nodes[j].real_track_id) <= 5):
                                    cost += self.extra_cost
                            else:
                                line_one = LineString([self.city_nodes[i].position, self.city_nodes[i].previous_pos])
                                line_two = LineString([self.city_nodes[j].position, self.city_nodes[j].previous_pos])
                                result = tsp.find_intersection(line_one, line_two)
                                if result["intersect_flag"]:
                                    dis_intersec_pt = ((self.city_nodes[i].position[0] - self.city_nodes[j].position[
                                        0]) ** 2 + \
                                                       (self.city_nodes[i].position[1] - self.city_nodes[j].position[
                                                           1]) ** 2) ** 0.5
                                    cost += 5 * dis_intersec_pt + 5
                                difference_vec_one = tuple(v1 - v2 for v1, v2 in zip(self.city_nodes[i].previous_pos,
                                                                                     self.city_nodes[i].position))
                                difference_vec_two = tuple(v1 - v2 for v1, v2 in zip(self.city_nodes[j].previous_pos,
                                                                                     self.city_nodes[j].position))
                                angle_diff = tsp.compute_angle(difference_vec_one, difference_vec_two)
                                if abs(angle_diff * 180 / math.pi) < 80  and (
                                        (self.city_nodes[i].position[0] - self.city_nodes[j].position[0]) ** 2 + \
                                        (self.city_nodes[i].position[1] - self.city_nodes[j].position[
                                            1]) ** 2) ** 0.5 < 3:
                                    cost += self.extra_cost
                            dis_martix[i][j] = -cost
                    else:
                        if self.city_nodes[i].node_type == NodeType.DUMMY or self.city_nodes[
                            j].node_type == NodeType.DUMMY:
                            dis_martix[i][j] = -self.dummy_cost
                        else:
                            print("the program cant enter here !")
        for i in range(size_citys):
            for j in range(i + 1, size_citys):
                dis_martix[i][j]  = dis_martix[j][i]
        print("compute dis matrix end !")

        return dis_martix

    def step(self,action):
        state = self.stops[-1]
        self.stops.append(action)
        # action 是从当前位置出发选择的下一个节点
        next_state = [action,self.stops]
        done = (len(self.stops) == (self.node_num))
        reward = self.dis_martix[state][next_state[0]]
        return next_state, reward, done

    def reset(self):
        self.stops = []
        first_stop = 1
        self.stops.append(first_stop)
        return [first_stop,self.stops]

    def render(self, return_img=False):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Show stops
        # end_x = self.node_pos[self.stops[0]][0]
        # end_y = self.node_pos[self.stops[0]][1]
        # self.x = np.concatenate([np.array([x[0] for x in self.node_pos]) , np.array([end_x])],axis=0)
        # self.y = np.concatenate([np.array([x[1] for x in self.node_pos]) , np.array([end_y])],axis=0)
        self.x = np.array([x[0] for x in self.node_pos])
        self.y = np.array([x[1] for x in self.node_pos])
        ax.scatter(self.x, self.y, c="red", s=50)

        # Show START
        xy = self.node_pos_dict.get(0)
        xytext = xy[0]*(1 + 0.1), xy[1]*(1 - 0.05)
        ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        x_coords = [self.x[i] for i in self.stops]  + [self.x[self.stops[0]]]
        y_coords = [self.y[i] for i in self.stops] + [self.y[self.stops[0]]]

        with open('coordinates.txt','w') as f :
            for x,y in zip(x_coords,y_coords):
                f.write(f"{x} {y}\n")  #每行一个坐标

        # Show itinerary
        # self.stops = list(range(self.node_num))+[0]
        ax.plot(self.x[self.stops+[self.stops[0]]], self.y[self.stops+[self.stops[0]]], c="blue", linewidth=1, linestyle="--")

        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()
    # def render(self, return_img=False):
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111)
    #
    #     # 初始化数据
    #     self.x = np.array([x[0] for x in self.node_pos])
    #     self.y = np.array([x[1] for x in self.node_pos])
    #
    #     # 绘制所有节点（初始为灰色）
    #     ax.scatter(self.x, self.y, c="gray", s=50)
    #
    #     # 初始化路径线（透明）
    #     path, = ax.plot([], [], c="blue", linewidth=1, linestyle="--", alpha=0)
    #
    #     # 高亮起点（初始为红色）
    #     start_idx = self.stops[0]
    #     start_scatter = ax.scatter(self.x[start_idx], self.y[start_idx], c="red", s=50)
    #
    #     # 存储所有高亮点对象
    #     scatter_objects = [start_scatter]
    #
    #     def update(frame):
    #         # 清除之前的高亮点（保留起点）
    #         for obj in scatter_objects[1:]:
    #             obj.remove()
    #         scatter_objects[1:] = []  # 只保留起点
    #
    #         # 当前路径段：从 stops[frame] 到 stops[frame+1]
    #         start = self.stops[frame]
    #         end = self.stops[frame + 1]
    #
    #         # 更新路径数据
    #         path.set_data([self.x[start], self.x[end]], [self.y[start], self.y[end]])
    #         path.set_alpha(1)  # 显示当前路径段
    #
    #         # 高亮当前终点节点
    #         end_scatter = ax.scatter(
    #             self.x[end],
    #             self.y[end],
    #             c="red",
    #             s=50,
    #             markeredgecolor="blue"
    #         )
    #         scatter_objects.append(end_scatter)
    #
    #         return path, *scatter_objects  # 返回所有需要更新的对象
    #
    #     # 创建动画（frames 必须是可迭代对象！）
    #     total_frames = len(self.stops) - 1
    #     anim = FuncAnimation(
    #         fig,
    #         update,
    #         frames=range(total_frames),  # 关键修正：传入 range(total_frames)
    #         interval=500,
    #         blit=True,  # 启用 blit 优化性能
    #         repeat=False
    #     )
    #
    #     if return_img:
    #         try:
    #             anim.save('animation.gif', writer='pillow', fps=2)
    #             return 'animation.gif'
    #         except ImportError:
    #             print("pillow 未安装，无法保存 GIF")
    #             plt.show()
    #             return None
    #     else:
    #         plt.show()  # 确保动画显示
    #         return anim  # 返回 anim 防止被垃圾回收

# tsp = tsp([10,10],8,100)
# print(tsp.node_pos)
# print(tsp.node_pos_dict)
# print(tsp.dist_dict.get((tsp.node_pos[0],tsp.node_pos[3])))
# print(tsp.dist_dict)