#显示L4耙地关键点的连接
import matplotlib.pyplot as plt

# with open('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/diagonal_behavior_plan/temp_0718.txt', 'r') as file:
#     data = file.read()
with open('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/diagonal_behavior_plan/all_key_path6.txt', 'r') as file:
    data = file.read()

# 将字符串数据转换为列表
lines = data.strip().split('\n')

# 解析数据点
points = [list(map(float, line.split())) for line in lines]

# 绘制线段
plt.figure(figsize=(10, 6))
for i in range(len(points) - 1):
    start_point = points[i]
    end_point = points[i + 1]
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], marker='o')

# # 在每个点旁边显示坐标
# for point in points:
#     plt.text(point[0], point[1], f'({point[0]:.2f}, {point[1]:.2f})')

# 设置图表标题和坐标轴标签
plt.title('Data Points Connection with Coordinates')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()