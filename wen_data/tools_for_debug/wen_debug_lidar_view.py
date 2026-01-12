"""
特徵 A：側視圖 (Side View) 必須是「平」的
這是最關鍵的指標。

✅ 正確 (Correct)：
藍色點雲像一條細長的水平線。
這條線緊緊貼在紅色參考線 (Z=−1.73) 上。
即使到了 40米、50米遠，它依然在紅線附近，沒有飄移。
就像你上傳的 kitti_MSNet3d_側面.png 那樣，像一塊扁平的鬆餅。

❌ 錯誤 1：起飛 (Flying / Ramp)
藍色點雲像飛機起飛一樣，距離越遠，高度越高。
原因：俯仰角 (Pitch) 修正不足，或者廣角畸變導致邊緣翹起。
MonoOcc 後果：遠處被判成「牆壁」或「人行道」，道路消失。

❌ 錯誤 2：香蕉球 (Banana / Curved)
近處貼合紅線，但遠處翹起來（變成 U 型或 J 型）。
原因：焦距 (Focal Length) 不對。這是廣角鏡頭最常見的問題。需要調整 focal-scale 來把彎曲的線拉直。

-----------

特徵 B：高度趨勢 (Trend) 必須是「水平」的
看圖表下方的折線圖（第3張子圖）。
✅ 正確：紅色的折線大致是一條水平直線，數值都在 -1.73 附近震盪。
❌ 錯誤：紅色的折線是斜線（往右上飆升）。

"""

import numpy as np
import matplotlib
# [新增] 強制使用 TkAgg 後端，確保在 Ubuntu/Linux 下視窗能順利彈出
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import argparse
import os

# 設定 MonoOcc / KITTI 的標準地面高度
TARGET_HEIGHT = -1.73
# save_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus/lidar_da3_v17"
# save_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar(MSNet3D)"
# bin_file = save_path + "/000125.bin"
# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/C330/lidar_da3/000090.bin"
bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_test/lidar_da3/1716262770251436490.bin"
# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar_da3_v6_autoscale_Tr/000000.bin"
save_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_test/lidar_da3"


def main():
    parser = argparse.ArgumentParser(description="Debug Pseudo-LiDAR Geometry")
    parser.add_argument('--bin', type=str, default=bin_file, help='Path to the .bin file')
    args = parser.parse_args()

    # 1. 讀取點雲
    # 假設格式為 (N, 3) 或 (N, 4) -> x, y, z, (intensity)
    points = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4)
    
    # 2. 篩選感興趣區域 (ROI) 以便觀察
    # 我們只看正前方 (左右 -5m ~ 5m)，這樣可以排除路邊建築的干擾，專心看路面
    mask_roi = (points[:, 1] > -5) & (points[:, 1] < 5) & \
               (points[:, 0] > 0) & (points[:, 0] < 60) # 看前方 0~60米
    pts = points[mask_roi]

    if len(pts) == 0:
        print("錯誤：點雲中沒有前方 60 米內的點。")
        return

    # 3. 繪製圖表
    # [修改] 加入視窗標題管理，方便辨識
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title(f'Pseudo-LiDAR Debugger: {os.path.basename(args.bin)}')

    # --- 子圖 1: 側面視圖 (Side View) ---
    # X軸: 前方距離 (Forward Distance)
    # Y軸: 高度 (Height Z)
    # 這是最重要的一張圖！用來檢查「起飛」或「下陷」
    plt.subplot(2, 2, 1)
    plt.scatter(pts[:, 0], pts[:, 2], s=0.5, c='blue', alpha=0.5, label='Points')
    
    
    # 畫出標準地面線
    plt.axhline(y=TARGET_HEIGHT, color='r', linestyle='-', linewidth=2, label='Ideal Ground (-1.73m)')
    plt.axhline(y=TARGET_HEIGHT + 2.0, color='g', linestyle='--', linewidth=1, label='Car Height (approx)')
    
    plt.title("1. Side View (Elevation Profile)\nCHECK: Is the blue line flat and on the red line?")
    plt.xlabel("Distance Forward (m)")
    plt.ylabel("Height Z (m)")
    plt.ylim(-5, 5) # 固定高度範圍以便比較
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # --- 子圖 2: 俯視圖 (Bird's Eye View) ---
    # X軸: 前方距離
    # Y軸: 左右距離
    # 用來檢查遮罩是否乾淨，以及是否有車輛輪廓
    plt.subplot(2, 2, 2)
    plt.scatter(pts[:, 0], pts[:, 1], s=0.5, c=pts[:, 2], cmap='viridis', alpha=0.5)
    plt.title("2. Bird's Eye View (Top Down)\nCHECK: Are there clear object clusters?")
    plt.xlabel("Distance Forward (m)")
    plt.ylabel("Left/Right (m)")
    plt.axis('equal')
    plt.grid(True)

    # --- 子圖 3: 距離與高度關係圖 (Pitch/Curve Analysis) ---
    # 這裡我們擬合一條趨勢線，看看斜率是多少
    plt.subplot(2, 1, 2)
    
    # 計算每 5 米的平均高度
    intervals = np.arange(0, 60, 5)
    mean_heights = []
    dists = []
    for i in range(len(intervals)-1):
        mask_d = (pts[:, 0] >= intervals[i]) & (pts[:, 0] < intervals[i+1])
        if np.sum(mask_d) > 10:
            h_avg = np.median(pts[mask_d, 2])
            mean_heights.append(h_avg)
            dists.append(intervals[i] + 2.5)
    
    plt.plot(dists, mean_heights, 'ro-', linewidth=2, label='Average Ground Height')
    plt.axhline(y=TARGET_HEIGHT, color='k', linestyle='--', label='Ideal Ground')
    
    plt.title("3. Ground Height Trend\nCHECK: This red line should be HORIZONTAL.")
    plt.xlabel("Distance Forward (m)")
    plt.ylabel("Average Height (m)")
    plt.ylim(-4, 2)
    plt.grid(True)
    plt.legend()

    # 儲存
    output_filename = os.path.join(save_path, os.path.basename(args.bin).replace('.bin', '_debug.png'))
    plt.tight_layout()
    # plt.savefig(output_filename)
    print(f"診斷圖表已儲存至: {output_filename}")
    print("請打開這張圖片進行分析。")
    
    # [確認] 確保視窗顯示，block=True 會讓程式暫停直到你關閉視窗
    print("正在開啟視窗... (請關閉視窗以結束程式)")
    plt.show(block=True)

if __name__ == "__main__":
    main()