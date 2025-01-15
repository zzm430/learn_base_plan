import numpy as np
from tqdm import tqdm
from Qlearing2 import Qlearning2
from Q_TSP import Tsp
import RL_utils

import matplotlib.pyplot as plt

chos = 1  # chos： 1 随机初始化地图； 0 导入固定地图
node_num = 20  # stop个数
map_size = [1,1]
end_rew = max(map_size)  # 结束奖励
num_episodes = 5e3  # 训练次数

env = Tsp(node_num = node_num,map_size= map_size,end_rew = end_rew)
agent = Qlearning2(ncol = node_num,
                   nrow = node_num,
                   epsilon = 0.5,
                   alpha = 0.2,
                   gamma = 0.926,
                   node_num = node_num,
                   final_epsilon = 1e-10)
return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state,reward,done = env.step(action)
                episode_return += reward
                agent.update(state,action,reward,next_state)
                state = next_state
            return_list.append(episode_return)
            if(i_episode +1 ) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

env.show_plot()

episodes_list = list(range(len(return_list)))

mv_return = RL_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('TSP'))
plt.show()
