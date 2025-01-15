import random

import numpy as np

class Qlearning2:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma,node_num,final_epsilon):
        self.Q_table = np.zeros(nrow , ncol)  # 初始化Q(s,a)表格
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.node_sum = node_num #节点总数
        self.final_epsilon = final_epsilon

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        q = np.copy(self.Q_table[state[0],:])
        q[state[1]] = -np.inf

        if(len(state[1] == self.node_sum)):
            action = state[1][0]
        elif np.random.random() < self.epsilon:
            action = random.choice([x for x in range(self.node_sum) if x not in state[1]])
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1[0]].max() - self.Q_table[s0[0], a0]
        self.Q_table[s0, a0] += self.alpha * td_error

        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay