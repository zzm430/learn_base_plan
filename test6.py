#功能:显示带有垄号信息的L2规划直线段
import matplotlib.pyplot as plt

# 从文件读取数据
# filename = '/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/level_two_plan/test_level_0724.txt'
filename = '/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/diagonal_behavior_plan/testcoarsmpath.txt'
with open(filename, 'r') as file:
    data = file.read()

# 将数据分割成行
lines = data.strip().split('\n')

# 初始化列表来存储垄号和对应的线段坐标点
ridge_segments = []

# 解析数据
i = 0
while i < len(lines) - 2:  # 确保至少还有两行数据
    ridge_number = int(lines[i].strip())
    i += 1
    # 读取并解析两组坐标点
    coords1 = list(map(float, lines[i].split()))
    i += 1
    coords2 = list(map(float, lines[i].split()))
    i += 1

    # 将垄号和两组坐标点存储在列表中
    ridge_segments.append((ridge_number, coords1, coords2))

# 绘制线段和坐标点，并在坐标点旁显示坐标值
fig, ax = plt.subplots()

# 遍历所有垄号和坐标点
for ridge_number, (x1, y1), (x2, y2) in ridge_segments:
    # 绘制垄号对应的线段
    ax.plot([x1, x2], [y1, y2], label=f'垄号 {ridge_number}')

    # 绘制坐标点，并在每个点旁显示坐标值
    ax.scatter([x1], [y1], color='red')  # 绘制第一个坐标点
    ax.text(x1, y1, f'({x1:.2f}, {y1:.2f})', fontsize=9, verticalalignment='bottom', horizontalalignment='right')

    ax.scatter([x2], [y2], color='blue')  # 绘制第二个坐标点
    ax.text(x2, y2, f'({x2:.2f}, {y2:.2f})', fontsize=9, verticalalignment='bottom', horizontalalignment='left')

# 设置图例
ax.legend()

# 设置坐标轴标题
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

# 显示网格
ax.grid(True)

# 显示图形
plt.title('Line Segments with Coordinate Values')
plt.show()