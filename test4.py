#显示耙地points_each_segment_初始关键点的连接

import matplotlib.pyplot as plt

# 假设你的数据存储在一个名为data.txt的文件中
with open('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/diagonal_block_plan/all_segments1.txt', 'r') as file:
    data = file.read()

# 将数据分割成行，并解析每一行
data_lines = data.strip().split('\n')
points_by_index = {}
index_order = []  # 用于记录索引出现的顺序

for line in data_lines:
    parts = line.split()
    index = int(parts[0])
    x = float(parts[1])
    y = float(parts[2])
    if index not in points_by_index:
        points_by_index[index] = []
        index_order.append(index)  # 记录索引出现的顺序
    points_by_index[index].append((x, y))

# 准备绘制
fig, ax = plt.subplots()

# 按照索引出现的顺序绘制每组索引对应的点，并用线段连接它们
for index in index_order:
    points = points_by_index[index]
    x_coords, y_coords = zip(*points)  # 解压坐标
    ax.plot(x_coords, y_coords, marker='o', label=f'Index {index}')

# 添加图例
ax.legend()

# 设置图表标题和坐标轴标签
ax.set_title('Data Visualization by Index Order')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')

# 显示图表
plt.show()
