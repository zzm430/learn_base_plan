import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

# 定义多边形顶点和方向向量
nodes = np.array([
    [0, 0],
    [2, 0],
    [3, 1],
    [1, 3],
    [-1, 1]
])
dir1 = np.array([1, 0])  # 方向向量 1
dir2 = np.array([0, 1])  # 方向向量 2
d = 0.2  # 等间隔
d = 0.2  # 等间隔

# 创建多边形对象
try:
    polygon = Polygon(nodes)
    print("多边形创建成功：", polygon)
except Exception as e:
    print("多边形创建失败：", e)

# 生成等间隔直线
def generate_lines(polygon, direction, spacing):
    min_x, min_y, max_x, max_y = polygon.bounds
    lines = []
    if direction[0] == 0:  # 垂直方向
        x = min_x
        while x <= max_x:
            line = LineString([(x, min_y), (x, max_y)])
            lines.append(line)
            x += spacing
    elif direction[1] == 0:  # 水平方向
        y = min_y
        while y <= max_y:
            line = LineString([(min_x, y), (max_x, y)])
            lines.append(line)
            y += spacing
    else:  # 斜方向
        t_min = -1000
        t_max = 1000
        start = np.array([min_x, min_y]) + t_min * direction
        end = np.array([max_x, max_y]) + t_max * direction
        line = LineString([start, end])
        lines.append(line)
    print(f"沿方向向量 {direction} 生成的直线数量：{len(lines)}")
    return lines

# 计算交点
def find_intersections(polygon, lines):
    intersections = set()
    for line in lines:
        if line.intersects(polygon):
            intersection = line.intersection(polygon)
            if intersection.geom_type == 'Point':
                intersections.add((intersection.x, intersection.y))
            elif intersection.geom_type == 'MultiPoint':
                for point in intersection.geoms:
                    intersections.add((point.x, point.y))
            elif intersection.geom_type == 'LineString':
                for point in intersection.coords:
                    intersections.add(tuple(point))
    print(f"找到的交点数量：{len(intersections)}")
    return intersections

# 生成沿 dir1 和 dir2 的直线
lines_dir1 = generate_lines(polygon, dir1, d)
lines_dir2 = generate_lines(polygon, dir2, d)

# 计算交点
intersections_dir1 = find_intersections(polygon, lines_dir1)
intersections_dir2 = find_intersections(polygon, lines_dir2)

# 合并交点
all_intersections = intersections_dir1.union(intersections_dir2)

# 输出结果
result = list(all_intersections)
print("生成的顶点集合：", result)

# 将结果保存到文本文件
def save_to_txt(filename, points):
    with open(filename, 'w') as file:
        for point in points:
            file.write(f"{point[0]}, {point[1]}\n")  # 每行存储一个点的坐标
    print(f"结果已保存到文件: {filename}")

# 调用保存函数
save_to_txt("output_points.txt", result)

# 可视化
def plot_polygon_and_points(polygon, points):
    # 绘制多边形
    x, y = polygon.exterior.xy
    plt.plot(x, y, label="Polygon", color="blue")

    # 绘制生成的点位
    if points:
        px, py = zip(*points)
        plt.scatter(px, py, color="red", label="Generated Points")

    # 设置图形属性
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon and Generated Points")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # 保持坐标轴比例一致
    plt.show()

# 调用可视化函数
plot_polygon_and_points(polygon, result)