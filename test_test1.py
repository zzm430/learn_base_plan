import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# 假设你已经有了points列表
# points = ...

with open('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/planning/neighbor_seeds_plan/first_fishnail_path_111.txt', 'r') as file:
    lines = file.readlines()

points = [tuple(map(float, line.strip().split())) for line in lines]

x_coords, y_coords = zip(*points)

plt.plot(x_coords, y_coords, 'o-', color='b')  # 绘制所有点

# 绘制第一个点为三角形
plt.plot(x_coords[0], y_coords[0], 'g^', markersize=10)

# 绘制最后一个点为方块
plt.plot(x_coords[-1], y_coords[-1], 'mv', markersize=10)

# 绘制箭头
for i in range(1, len(points)):
    x_start, y_start = points[i - 1]
    x_end, y_end = points[i]
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='->', mutation_scale=20,
        ec='k',  # 箭头边缘颜色
        fc='k'  # 箭头填充颜色
    )
    plt.gca().add_patch(arrow)

# 显示所有点的坐标
for i, (x, y) in enumerate(points):
    plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=12, ha='right' if x < x_coords[-1] else 'left')

plt.title('Connected Points with Arrows and Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()