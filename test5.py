#显示L4耙地段内连接和段间连接用不同的颜色展示
import matplotlib.pyplot as plt

# 假设读取的文件路径
file_path = '/home/aiforce/harrow_project/SmartFarmWorkspace/cmake-build-debug/src/planning/diagonal_behavior_plan/allPoints.txt'

# 定义颜色
segment_color = 'blue'  # 段内连接线段的颜色
inter_segment_color = 'green'  # 段间连接线段的颜色

# 读取文件并按空行分割成段
with open(file_path, 'r') as file:
    all_data = file.read().strip()

# 检查文件是否包含数据
if not all_data:
    print("文件为空")
    exit()

# 按行分割数据点
lines = all_data.split('\n')

# 初始化上一段的最后一个点
last_point_of_previous_segment = None

# 开始绘制
plt.figure(figsize=(10, 6))

for line in lines:
    if not line.strip():  # 空行，表示新的一段开始
        if last_point_of_previous_segment is not None:  # 如果不是第一段，绘制段间连接线段
            current_point = [float(x) for x in lines[lines.index(line)].split()]
            plt.plot([last_point_of_previous_segment[0], current_point[0]],
                     [last_point_of_previous_segment[1], current_point[1]],
                     color=inter_segment_color, lw=1)
        continue

    # 解析当前行的数据点
    current_point = [float(x) for x in line.split()]

    # 绘制当前点
    plt.plot(current_point[0], current_point[1], 'ro', markersize=2)

    # 如果上一个点存在，绘制段内连接线段
    if last_point_of_previous_segment is not None:
        plt.plot([last_point_of_previous_segment[0], current_point[0]],
                 [last_point_of_previous_segment[1], current_point[1]],
                 color=segment_color, lw=1)

    # 更新上一个点
    last_point_of_previous_segment = current_point

# 设置图表标题和坐标轴标签
plt.title('Data Points Connection with Coordinates')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()