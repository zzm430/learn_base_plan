#显示L4耙地中的几何算法路径
#
# import matplotlib.pyplot as plt
#
# with open('/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/diagonal_behavior_plan/geopathfile.txt', 'r') as file:
#     data = file.read()
#
# # 将字符串数据转换为列表
# lines = data.strip().split('\n')
#
# # 解析数据点
# points = [list(map(float, line.split())) for line in lines]
#
# # 绘制线段
# plt.figure(figsize=(10, 6))
# for i in range(len(points) - 1):
#     start_point = points[i]
#     end_point = points[i + 1]
#     plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], marker='o')
#
# # # 在每个点旁边显示坐标
# # for point in points:
# #     plt.text(point[0], point[1], f'({point[0]:.2f}, {point[1]:.2f})')
#
# # 设置图表标题和坐标轴标签
# plt.title('Data Points Connection with Coordinates')
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
#
# # 显示网格
# plt.grid(True)



# 显示图表
# plt.show()


import matplotlib.pyplot as plt

# 读取第一个文件的数据
with open('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/executor/planning/optizepath.txt', 'r') as file:
    data1 = file.read()

lines1 = data1.strip().split('\n')
points1 = [list(map(float, line.split())) for line in lines1]

# 读取第二个文件的数据
with open('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/executor/planning/pts_of_each_segment.txt', 'r') as file:
    data2 = file.read()

lines2 = data2.strip().split('\n')
points2 = [list(map(float, line.split())) for line in lines2]

# 绘制第一个文件的数据点
plt.figure(figsize=(10, 6))
for i in range(len(points1) - 1):
    start_point = points1[i]
    end_point = points1[i + 1]
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], label='Path 1', marker='o')

# 绘制第二个文件的数据点
for i in range(len(points2) - 1):
    start_point = points2[i]
    end_point = points2[i + 1]
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], label='Path 2', linestyle='--', marker='x')

# 设置图表标题和坐标轴标签
plt.title('Data Points Connection with Coordinates')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()