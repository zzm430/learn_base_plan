import random
import numpy as np
from itertools import combinations
import copy
import matplotlib.pyplot as plt
from sympy.physics.units import radian


class Tsp:
    def __init__(self, map_size, node_num, end_rew):
        self.node_num = node_num
        self.map_size = map_size
        self.end_rew = end_rew
        self.node_pos = self.generate_node_pos()  # 随机生成node_num个站点
        self.node_pos_dict = {cnt: pos for cnt, pos in zip(range(len(self.node_pos)), self.node_pos)}
        self.dist_dict = self.cal_dist(node_pos=self.node_pos)
        self.stops = []  # 按顺序记录依次经过了哪些stop
        self.rew_mat = self.reward_matrix(node_num)


    def generate_node_pos(self):
        origin_pts = set()
        while(len(origin_pts) < self.node_num):
            x = random.uniform(0,self.map_size[0])
            y = random.uniform(0,self.map_size[1])
            origin_pts.add((x,y))

        origin_pts_list = list(origin_pts)
        return origin_pts_list

    def  cal_dist(self,node_pos):
        distances_dict = {}
        for pair in combinations(node_pos,2):
            distance = self.calculate_dis(pair[0],pair[1]);
            distances_dict[pair] = distance
            distances_dict[pair[1],pair[0]] = distance
        return distances_dict   

    def calculate_dis(self,point1,point2):
        return  np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


    def rewward_matrix(self,node_sum):
        rew_matrix = np.zeros([node_sum,node_sum+1])
        for row in range(node_sum):
            for col in range(node_sum):
                if row == col:
                    rew_matrix[row][col] = np.NAN
                else:
                    rew_matrix[row][col] = -self.dist_dict.get(self.node_pos[row],self.node_pos[col])
                    
        return rew_matrix                
    
    def show_plot(self):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

        self.x = np.array([x[0] for x in self.node_pos])
        self.y = np.array([x[1] for x in self.node_pos])
        ax.scatter(self.x, self.y, c="red", s=50)

        xy = self.node_pos_dict.get(0)
        xytext = xy[0] * (1 + 0.1), xy[1] * (1 - 0.05)
        ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        ax.plot(self.x[self.stops + [self.stops[0]]], self.y[self.stops + [self.stops[0]]], c="blue", linewidth=1,
                linestyle="--")

        plt.show()

    def reset(self):
        self.stops = []
        first_position = np.random.randint(self.node_num)
        self.stops.append(first_position)
        return [first_position,self.stops]

    def step(self,action):
        state = self.stops[-1]
        self.stops.append(action)
        next_state = [action,self.stops]

        done = (len(self.stops) == (self.node_num + 1))

        reward = self.rew_mat[state][next_state[0]]

        return  next_state,reward,done