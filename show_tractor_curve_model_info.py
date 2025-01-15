import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

tractorHeadPtsStream111 = np.loadtxt('/home/aiforce/new_harrow_project_1107/SmartFarmWorkspace/cmake-build-debug/src/planning/neighbor_seeds_plan/tractor_head_trailer_model.txt')

fig, ax = plt.subplots()


for m in range(len(tractorHeadPtsStream111) - 1):
    if m % 2 == 0:
        coords11 = [(tractorHeadPtsStream111[m][j], tractorHeadPtsStream111[m + 1][j]) for j in range(4)]
        coords12 = [(tractorHeadPtsStream111[m][j], tractorHeadPtsStream111[m + 1][j]) for j in range(4, 8)]
        polygon1 = Polygon(coords12, closed=True, fill=False, edgecolor='green', alpha=0.9, linewidth=1)
        polygonf1 = Polygon(coords11, closed=True, fill=False, edgecolor='gray', alpha=0.9, linewidth=1)
        # 设置透明度为 0.5
        ax.add_patch(polygon1)
        ax.add_patch(polygonf1)

plt.show()