import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#可用于测试入垄作业是否对齐 2024/12/11
# 加载数据
tractorHeadPtsStream111 = np.loadtxt('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/planning/neighbor_seeds_plan/tractor_head_trailer_model.txt')

fig, ax = plt.subplots()

# 计算所有坐标的最小和最大值

pointA = (-15.934, -138.176)  # 替换x1, y1为实际的坐标值
pointB = ( -7.7, 136.055)  # 替换x2, y2为实际的坐标值

pointC = (-15.934, -138.176)  # 替换x1, y1为实际的坐标值
pointD = ( -7.7, 136.055)  # 替换x2, y2为实际的坐标值

for m in range(len(tractorHeadPtsStream111) - 1):
    if m % 2 == 0:
        coords11 = [(tractorHeadPtsStream111[m][j], tractorHeadPtsStream111[m + 1][j]) for j in range(4)]
        coords12 = [(tractorHeadPtsStream111[m][j], tractorHeadPtsStream111[m + 1][j]) for j in range(4, 8)]
        polygon1 = Polygon(coords12, closed=True, fill=False, edgecolor='green', alpha=0.8, linewidth=1)
        polygonf1 = Polygon(coords11, closed=True, fill=False,edgecolor='gray', alpha=0.8, linewidth=1)
        ax.add_patch(polygon1)
        ax.add_patch(polygonf1)

# 设置坐标轴范围
# 设置坐标轴等比例
ax.axis('equal')
ax.plot([pointA[0], pointB[0]], [pointA[1], pointB[1]], marker='o')
ax.plot([pointC[0], pointD[0]], [pointC[1], pointD[1]], marker='o')
# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()