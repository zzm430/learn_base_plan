import matplotlib.pyplot as plt

# 读取TXT文件中的数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip()

    # 从字符串中提取坐标列表
    points = eval(data)

    # 提取x和y坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    return x_coords, y_coords

# 主函数
def main():
    # 替换为你的TXT文件路径
    file_path = '/home/aiforce/test.txt'

    # 从TXT文件中读取数据
    x_coords, y_coords = read_data_from_txt(file_path)

    # 绘制点
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')  # 使用'o'标记每个点，并连接线

    # 在每个点的附近显示坐标
    for x, y in zip(x_coords, y_coords):
        plt.text(x, y, f'({x:.4f}, {y:.4f})', fontsize=8, ha='right')

    plt.title('Data Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

# 运行主函数
if __name__ == "__main__":
    main()