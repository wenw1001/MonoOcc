import numpy as np
import matplotlib.pyplot as plt

# 1. 設定檔案路徑
# npy_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/depth(MSNet3D)/000070.npy"
# npy_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/depth_da3_v6_autoscale_Tr/000095.npy"
# npy_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_test_kitti/depth_da3/1716262770129963065.npy"
npy_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day/depth_da3/1718183083816408675.npy"

# npy_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/C330/depth_da3/000090.npy"

# 2. 讀取深度數據
try:
    depth_data = np.load(npy_path)
    print(f"成功讀取: {npy_path}")
    print(f"數據形狀: {depth_data.shape}")
    print(f"最大深度: {np.nanmax(depth_data):.2f}m, 最小深度: {np.nanmin(depth_data):.2f}m")
except FileNotFoundError:
    print(f"找不到檔案: {npy_path}")
    exit()

# 3. 設定繪圖
fig, ax = plt.subplots(figsize=(10, 6))

# 使用 inferno 色調顯示深度 (你可以改用 'viridis', 'plasma', 'gray' 等)
# 我們直接畫 .npy 的數據，這樣顏色對應才是準確的
img = ax.imshow(depth_data, cmap='inferno')
plt.colorbar(img, label='Depth (meters)')
ax.set_title("Hover mouse to see depth value")

# 4. 定義滑鼠滑動事件
def on_move(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        # 邊界檢查
        if 0 <= x < depth_data.shape[1] and 0 <= y < depth_data.shape[0]:
            val = depth_data[y, x]
            # 更新標題顯示座標與深度
            ax.set_title(f"Pos: ({x}, {y}) | Depth: {val:.4f} m")
            fig.canvas.draw_idle() # 更新畫布

# 綁定事件
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()