# #展示L2规划的路径信息
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig, ax = plt.subplots()
# lineshow = np.loadtxt('/home/aiforce/stoage_ceshi_txt/temp.txt')
#
#
# # 绘制线段
# for i in range(0, len(lineshow[0])-1, 2):
#     x = [lineshow[0][i], lineshow[0][i+1]]
#     y = [lineshow[1][i], lineshow[1][i+1]]
#     plt.plot(x, y)
#     plt.text(lineshow[0][i], lineshow[1][i], f"({lineshow[0][i]}, {lineshow[1][i]})")
#
# ax.set_xlabel('x label')
# ax.set_ylabel('y label')
# ax.set_title('Simple Plot')
# ax.set_aspect('equal')
# # ax.autoscale()
# ax.legend()
#
# # plt.xlim(0, 700) # 横坐标显示范围为0到6
# # plt.ylim(0, 1500)
#
# # mng = plt.get_current_fig_manager()
# # mng.full_screen_toggle()
# fig.tight_layout()
#
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

#L2 n+m 垄处理
# lineshow = np.loadtxt('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/level_two_nplusm_plan/temp_level_two_nplusm.txt')
lineshow = np.loadtxt('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/level_two_nplusm_plan/temp_level_two_nplusm.txt')
# 绘制线段
for i in range(0, len(lineshow[0]) - 1, 2):
    x = [lineshow[0][i], lineshow[0][i + 1]]
    y = [lineshow[1][i], lineshow[1][i + 1]]
    plt.plot(x, y)

    # 显示起始点坐标
    plt.text(lineshow[0][i], lineshow[1][i], f"({lineshow[0][i]}, {lineshow[1][i]})")

    # 显示终点坐标
    plt.text(lineshow[0][i + 1], lineshow[1][i + 1], f"({lineshow[0][i + 1]}, {lineshow[1][i + 1]})")

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title('Simple Plot')
ax.set_aspect('equal', 'box')  # 确保坐标轴的比例为1:1
# ax.autoscale()
ax.legend()

# plt.xlim(0, 700) # 横坐标显示范围为0到6
# plt.ylim(0, 1500)

# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
fig.tight_layout()

plt.show()