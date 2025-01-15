import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# 假设你已经有了points列表
# points = ...

with open('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/planning/neighbor_seeds_plan/first_fishnail_path_111.txt', 'r') as file:
    lines = file.readlines()

points = [tuple(map(float, line.strip().split())) for line in lines]

x_coords, y_coords = zip(*points)

plt.plot(x_coords[1:-1], y_coords[1:-1], 'o-', color='b')
# 绘制所有点，除了第一个和最后一个点
plt.plot(x_coords[1:-1], y_coords[1:-1], 'o-', color='b')

# 绘制第一个点为三角形
plt.plot(x_coords[0], y_coords[0], 'g^', markersize=10)

# 绘制最后一个点为方块
plt.plot(x_coords[-1], y_coords[-1], 'mv', markersize=10)

# 绘制箭头
for i in range(1, len(points)):
    # 计算箭头的起点和终点
    x_start, y_start = points[i - 1]
    x_end, y_end = points[i]

    # 创建箭头
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='->', mutation_scale=20,
        ec='k',  # 箭头边缘颜色
        fc='k'  # 箭头填充颜色
    )

    # 将箭头添加到图表中
    plt.gca().add_patch(arrow)

# 显示所有点的坐标
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=12, ha='right' if i < len(points) - 1 else 'left')

# 设置图表标题和坐标轴标签
plt.title('Connected Points with Arrows')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()