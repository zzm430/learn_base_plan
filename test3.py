import matplotlib.pyplot as plt

with open('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/level_two_plan/temp_level_two.txt', 'r') as file:
    data = file.read()

# 将字符串数据转换为列表
lines = data.strip().split('\n')

# 解析数据点
points = [list(map(float, line.split())) for line in lines]

# 绘制线段
plt.figure(figsize=(10, 6))
for i in range(0, len(points), 2):  # 步长设置为2
    start_point = points[i]
    # 检查是否有足够的点来绘制线段
    if i + 1 < len(points):
        end_point = points[i + 1]
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], marker='o', label='Line Segment')
        plt.text(start_point[0], start_point[1], f'({start_point[0]:.2f}, {start_point[1]:.2f})', fontsize=8)
        plt.text(end_point[0], end_point[1], f'({end_point[0]:.2f}, {end_point[1]:.2f})', fontsize=8)

# 设置图表标题和坐标轴标签
plt.title('Data Points Connection')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# 显示图表
plt.grid(True)
plt.legend()
plt.show()