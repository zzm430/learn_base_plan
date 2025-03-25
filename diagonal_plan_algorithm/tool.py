#动画绘制出点位的路径流
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 1. 读取数据
with open('coordinates.txt', 'r') as f:
    lines = [list(map(float, line.strip().split())) for line in f if line.strip()]

x, y = zip(*lines)  # 解压为x和y坐标列表

# 2. 创建画布
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(min(x) - 1, max(x) + 1)
ax.set_ylim(min(y) - 1, max(y) + 1)
ax.set_title('Dynamic Points Connection')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)

# 初始化空线条和点
line, = ax.plot([], [], 'b-', linewidth=1)
point, = ax.plot([], [], 'bo', markersize=4)
annotations = []


# 3. 动画更新函数
def update(frame):
    # 更新线条数据（连接所有已经过的点）
    line.set_data(x[:frame + 1], y[:frame + 1])

    # 更新当前点位置
    point.set_data([x[frame]], [y[frame]])

    # 添加/更新标注
    if len(annotations) <= frame:
        anno = ax.text(x[frame], y[frame], str(frame),
                       fontsize=8, color='red')
        annotations.append(anno)

    return line, point, *annotations


# 4. 创建动画
ani = FuncAnimation(
    fig, update,
    frames=len(x),
    interval=200,  # 每帧间隔(ms)
    blit=True,
    repeat=False
)

plt.tight_layout()
plt.show()

# 如需保存动画（取消下面注释）
ani.save('animation.mp4', writer='ffmpeg', fps=10)