import matplotlib.pyplot as plt

# 读取txt文件中的数据
with open('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/planning/common/avoid_obstacle_algorithm/test/curve_test.txt', 'r') as file:
    data = file.readline().strip().split()

# 将数据转换为数值类型
data = [float(x) for x in data]

# 创建折线图
plt.plot(data,color='g',markerfacecolor='green',marker='o',label='keypoints data')

# 显示每个数据点的标签
for i, d in enumerate(data):
    plt.text(i, d, str(d), ha='center', va='bottom', fontsize=10)

# 添加标题和坐标轴标签
plt.title('Data Line Chart')
plt.xlabel('Index')
plt.ylabel('Value')

# 显示折线图
plt.show()