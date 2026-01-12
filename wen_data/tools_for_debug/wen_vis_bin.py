"""
這是一個用於可視化 LiDAR 點雲的臨時腳本。
可以用來檢查 preprocess_one_click 將深度轉點雲後
點雲的分佈和高度信息。
"""

import open3d as o3d
import numpy as np
import argparse

# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/C330/lidar_da3/000090.bin"
bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_rain_test_kitti/lidar_da3_v2/1716262770251436490.bin"
# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day_test_kitti/lidar_da3_v1/1718183083816408675.bin"
# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar_da3_v5_autoscale_Tr/000095.bin"
# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar_da3_v10/000095.bin"
# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar(MSNet3D)/000000.bin"

# bin_file = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00/lidar_da3/000000.bin"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin', default=bin_file, help='Path to .bin file')
    args = parser.parse_args()

    # 讀取 .bin (x, y, z)
    # 假設 MonoOcc 格式是 N x 4 (x, y, z, intensity) 或 N x 3
    raw_data = np.fromfile(args.bin, dtype=np.float32)
    # 嘗試 reshape，通常是 Nx4
    try:
        points = raw_data.reshape(-1, 4)[:, :3]
    except:
        points = raw_data.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 建立座標軸 (紅X, 綠Y, 藍Z) - 原點在車體中心
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # 建立一個參考地面網格 (Z = -1.73)
    # 畫一個大框框代表理想地面高度
    lines = []  
    colors = []
    # 畫一個 20x20 米的網格，高度在 -1.73
    grid_z = -1.73
    min_x, max_x = 0, 40
    min_y, max_y = -10, 10
    
    # 畫框框的四個角
    corners = [
        [min_x, min_y, grid_z], [max_x, min_y, grid_z],
        [max_x, max_y, grid_z], [min_x, max_y, grid_z]
    ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([1, 0, 0]) # 紅色框框代表 MonoOcc 的理想地面

    print("紅色框框是標準地面高度 (-1.73m)。")
    print("請檢查點雲是否與紅框重疊？是否水平？")
    
    o3d.visualization.draw_geometries([pcd, axis, line_set], 
                                      window_name="Check Point Cloud",
                                      width=1980, height=1280)

if __name__ == "__main__":
    main()