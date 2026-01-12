"""
conda activate da3

python /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click.py \
    --image-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00/image_2 \
    --calib-path /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00/calib.txt \
    --save-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00 \
    --save-depth \
    --save-depth-vis \
    --save-lidar \
    --compress



python /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click.py \
    --image-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus/image_2 \
    --calib-path /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus/calib.txt \
    --save-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus \
    --save-depth \
    --save-depth-vis \
    --save-lidar \
    --manual-pitch -2.0 \
    --compress

    --manual-height 3.0 # 設定了這個參數後，auto-scale 會被忽略
    --manual-yaw 5.0  <-- 新增：正值向左轉，負值向右轉

此腳本實現了一鍵式的圖像到體素預處理流程，包含以下步驟：
1. 使用 Depth Anything 3 (DA3) 模型從輸入圖像生成深度圖。
2. 將深度圖轉換為 LiDAR 點雲座標。
3. 將點雲體素化為二進位 voxel 格式。

    
"""

import argparse
import os
import glob
import numpy as np
import torch
import sys
import cv2
from tqdm import tqdm

mapping_lib_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mapping"

if mapping_lib_path not in sys.path:
    sys.path.insert(0, mapping_lib_path)

try:
    import mapping
    print(f"成功載入 mapping 模組: {mapping.__file__}")
except ImportError as e:
    print(f"無法載入 mapping，請確認路徑是否正確: {mapping_lib_path}")
    raise e
import mapping


# 嘗試匯入 Depth Anything 3
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("錯誤: 找不到 'depth_anything_3'。請確認你已啟動安裝了 DA3 的環境")
    sys.exit(1)

# --- KITTI 參數 ---
# KITTI 相機高度約為 1.73 米 (相機座標系 Y 軸向下，地面在 +1.73m 處)
KITTI_CAMERA_HEIGHT = 1.55 

DEFAULT_CALIB_K = np.array([
    [721.5377, 0.0, 609.5593],
    [0.0, 721.5377, 172.8540],
    [0.0, 0.0, 1.0]
])

# 修正後的標準 KITTI Tr (用於 fallback)
DEFAULT_TR_VELO_TO_CAM = np.array([
    [0.0, -1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0,  0.0],
    [1.0,  0.0,  0.0,  0.0],
    [0.0,  0.0,  0.0,  1.0]
])

DEFAULT_CALIB = {
    'P2': np.hstack((DEFAULT_CALIB_K, np.zeros((3, 1)))),
    'R0_rect': np.eye(4),
    'Tr_velo_to_cam': DEFAULT_TR_VELO_TO_CAM[:3, :]
}

PC_RANGE = np.array([0, -25.6, -2.0, 51.2, 25.6, 4.4])
# 注意：這裡 voxel size 配合你的 grid size (256, 256, 32)
# X: 0~51.2 / 256 = 0.2
# Y: -25.6~25.6 / 256 = 0.2
# Z: -2.0~4.4 / 32 = 0.2
VOXEL_SIZE = np.array([0.2, 0.2, 0.2])
GRID_SIZE = np.array([256, 256, 32])


def parse_args():
    parser = argparse.ArgumentParser(description='One-click preprocess: Image -> Depth -> Lidar -> Voxel')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image-path', type=str, help='Path to a single image file')
    group.add_argument('--image-dir', type=str, help='Path to a directory containing images')
    parser.add_argument('--calib-path', type=str, required=True, help='Path to KITTI calib.txt')
    parser.add_argument('--save-dir', type=str, help='Custom output directory.')
    parser.add_argument('--model-type', type=str, default='depth-anything/DA3NESTED-GIANT-LARGE')
    parser.add_argument('--max-depth', type=float, default=80.0)
    
    # --- 新增/修改參數 ---
    parser.add_argument('--auto-scale', action='store_true', default=True, help='自動根據地面高度計算 Scale (預設開啟)')
    parser.add_argument('--fixed-scale', type=float, default=1.0, help='如果關閉 auto-scale，則使用此固定數值')

    parser.add_argument('--manual-pitch', type=float, default=0.0, 
                        help='手動調整 LiDAR 俯仰角 (度)。正值=車頭向下壓(路面變低)，負值=車頭抬高。')
    parser.add_argument('--manual-yaw', type=float, default=0.0, 
                        help='手動調整 LiDAR 偏航角 (Yaw/水平旋轉, 度)。正值=向左轉(逆時針)，負值=向右轉(順時針)。')
    parser.add_argument('--manual-height', type=float, default=None, 
                        help='手動設定相機高度 (米)。若設定此值，將強制覆蓋 Auto Scale。公式: Scale = 1.73 / manual_height')
    parser.add_argument('--focal-scale', type=float, default=1.0, 
                        help='焦距縮放係數。廣角鏡頭請設 < 1.0 (例如 0.8)，可讓視野變廣並改善地面彎曲。')
    parser.add_argument('--z-stretch', type=float, default=1.0, 
                        help='Z軸(高度)拉伸係數。設為 > 1.0 (如 2.0) 可以把被壓扁的車子拉高，且不移動地板位置。')
    
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--save-depth', action='store_true')
    parser.add_argument('--save-depth-vis', action='store_true')
    parser.add_argument('--save-lidar', action='store_true')
    return parser.parse_args()

def pack(array):
    array = array.reshape((-1)).astype(np.uint8)
    compressed = (array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | 
                  array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8])
    return np.array(compressed, dtype=np.uint8)

def read_calib_file(filepath):
    """讀取 KITTI calib 檔案 (相容 Odometry/Raw)"""
    if not os.path.exists(filepath): return DEFAULT_CALIB
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' not in line: continue
            key, value = line.split(':', 1)
            try: data[key] = np.array([float(x) for x in value.split()])
            except ValueError: pass
    
    calib = {}
    calib['P2'] = data['P2'].reshape(3, 4) if 'P2' in data else DEFAULT_CALIB['P2']
    
    if 'R0_rect' in data:
        R0 = data['R0_rect'].reshape(3, 3)
        R0_rect = np.eye(4); R0_rect[:3, :3] = R0; calib['R0_rect'] = R0_rect
    else: calib['R0_rect'] = np.eye(4)
    
    if 'Tr_velo_to_cam' in data: Tr = data['Tr_velo_to_cam'].reshape(3, 4)
    elif 'Tr' in data: Tr = data['Tr'].reshape(3, 4)
    else: Tr = DEFAULT_CALIB['Tr_velo_to_cam'][:3, :]
    
    Tr_mat = np.eye(4); Tr_mat[:3, :4] = Tr; calib['Tr_velo_to_cam'] = Tr_mat
    return calib

def step1_predict_depth(model, image_path):
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"無法讀取: {image_path}")
    h, w = img.shape[:2]
    pred = model.inference([image_path])
    depth = pred.depth[0]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth

def estimate_scale_from_ground(depth_map, calib):
    """
    自動計算 Scale:
    假設圖片底部中央區域是平坦地面，且相機高度應為 1.73m。
    """
    h, w = depth_map.shape
    
    # 1. 定義 ROI: 圖片底部中央 (通常是路面)
    # 範圍: 高度 85%~95%, 寬度 45%~65%
    y_min, y_max = int(h * 0.85), int(h * 0.95)
    x_min, x_max = int(w * 0.45), int(w * 0.65)
    
    # 取出深度值
    roi_depth = depth_map[y_min:y_max, x_min:x_max]
    print(f"  [Auto Scale] 使用 ROI 範圍: Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")
    print(f"    ROI 深度範圍: {roi_depth.min():.3f} ~ {roi_depth.max():.3f} m")
    print(f"    ROI 深度平均: {roi_depth.mean():.3f} m")
    


    # 建立網格座標 (u, v)
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    
    # 取得內參
    P2 = calib['P2']
    fx, fy, cy = P2[0, 0], P2[1, 1], P2[1, 2]
    
    # 2. 反投影算出相機座標下的 Y (高度)
    # Y_cam = (v - cy) * Z / fy
    # 這裡 Z 就是 roi_depth
    Y_cam = (yy - cy) * roi_depth / fy
    print(f"    ROI 高度範圍: {Y_cam.min():.3f} ~ {Y_cam.max():.3f} m")
    print(f"    ROI 高度平均: {Y_cam.mean():.3f} m")
    
    # 3. 計算預測的平均高度
    # 在 KITTI 相機座標系中，Y 軸向下，所以地面高度是正值
    valid_mask = (roi_depth > 0) & (roi_depth < 40.0) # 只看 40米內的地面，太遠不準
    valid_Y = Y_cam[valid_mask]
    
    if len(valid_Y) < 100:
        print("警告: 地面點不足，使用預設 Scale 1.0")
        return 1.0
        
    # median_h = np.median(valid_Y)

    # --- 強力過濾雜訊 ---
    p25, p75 = np.percentile(valid_Y, [25, 75])
    strict_mask = (valid_Y >= p25) & (valid_Y <= p75)
    clean_Y = valid_Y[strict_mask]
    
    if len(clean_Y) < 10: 
        print("過濾後地面點不足，使用過濾前的點雲")
        median_h = np.median(valid_Y)
    else:
        median_h = np.median(clean_Y)
    
    # 4. 計算 Scale
    # 真實高度 / 預測高度
    # 目標是讓 median_h 變成 KITTI_CAMERA_HEIGHT (1.73)
    # TARGET_HEIGHT = 1.9  # 從 1.73 改成 1.60 (越小，Scale 會越小，地板拉越高)
    # TARGET_HEIGHT = KITTI_CAMERA_HEIGHT
    scale = KITTI_CAMERA_HEIGHT / median_h

    # 額外的經驗修正因子 (Safety Factor)
    # 針對 DA3 容易估太遠的特性，再強制縮小一點 (例如 0.9 倍)
    # 這樣如果算出 0.8，就會變成 0.72，接近你的 0.65 經驗值，kitti最佳是0.9
    SAFETY_FACTOR = 0.9 
    
    scale = scale * SAFETY_FACTOR
    
    # 5. 安全限制 (避免算出太離譜的數值)
    # scale = np.clip(scale, 0.4, 1.4)
    
    print(f"  [Auto Scale] 預測地面高度: {median_h:.3f} m, 計算 Scale: {scale:.4f}")
    return scale

def save_depth_vis(depth_map, save_path):
    d_min, d_max = depth_map.min(), depth_map.max()
    d_norm = (depth_map - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_map)
    d_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(save_path, d_vis)

def step2_depth_to_pointcloud(depth_map, calib, max_depth, scale, focal_scale):
    """加入 Scale 參數的投影"""
    
    # 應用 Scale
    depth_map = depth_map * scale
    
    rows, cols = depth_map.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))

    # # [建議新增] 簡單的梯度過濾，去除邊緣拖尾，但會導致lidar變稀疏一點
    # # 計算 X 和 Y 方向的梯度
    # grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    # grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # # 設定閾值，例如深度變化太劇烈的地方視為無效
    # # 這個閾值需要根據數據調整，例如 2.0 或 5.0
    # edge_mask = grad_mag < 5.0

    mask = (depth_map > 0) & (depth_map < max_depth) # & edge_mask
    depth = depth_map[mask]
    u, v = c[mask], r[mask]
    
    P2 = calib['P2']
    # fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    # 廣角鏡頭: focal_scale < 1.0 (縮短焦距，擴大 FOV)
    fx = P2[0, 0] * focal_scale
    fy = P2[1, 1] * focal_scale
    cx, cy = P2[0, 2], P2[1, 2]
    tx = P2[0, 3] / (-fx)
    
    z = depth
    x = (u - cx) * z / fx + tx
    y = (v - cy) * z / fy
    
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
    
    # 轉換到 LiDAR 座標
    R0_inv = np.linalg.inv(calib['R0_rect'])
    Tr_inv = np.linalg.inv(calib['Tr_velo_to_cam'])
    points_lidar = (Tr_inv @ R0_inv @ points_cam)[:3, :].T
    

    # # 如果需要過濾掉離地太高 (例如 > 5米) 的雜訊點
    # valid_height = points_lidar[:, 2] < 5.0  # KITTI LiDAR 在地面約 -1.73m，車頂約 0m，5m 很高了
    # points_lidar = points_lidar[valid_height]
    
    return points_lidar

# def step3_pointcloud_to_voxel(points, pc_range, voxel_size, grid_size):
#     keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) & \
#            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) & \
#            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
#     points = points[keep]
#     if points.shape[0] == 0: return np.zeros(tuple(grid_size), dtype=np.float32)
#     coords = np.floor((points - pc_range[:3]) / voxel_size).astype(np.int32)
#     unique_coords = np.unique(coords, axis=0)
#     voxel = np.zeros(tuple(grid_size), dtype=np.float32)
#     x, y, z = unique_coords.T
#     valid = (x>=0) & (x<grid_size[0]) & (y>=0) & (y<grid_size[1]) & (z>=0) & (z<grid_size[2])
#     voxel[x[valid], y[valid], z[valid]] = 1.0
#     return voxel

def apply_manual_pitch(points, pitch_deg):
    """
    [NEW] 對 LiDAR 座標系點雲進行俯仰角 (Pitch) 旋轉
    KITTI LiDAR: x=前, y=左, z=上
    旋轉軸: y軸 (左右)
    """
    if pitch_deg == 0.0:
        return points

    theta = np.radians(pitch_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # 繞 Y 軸旋轉矩陣
    # x' = x*cos + z*sin
    # z' = -x*sin + z*cos
    
    x = points[:, 0]
    z = points[:, 2]
    
    points[:, 0] = x * c + z * s
    points[:, 2] = -x * s + z * c
    
    return points

def apply_manual_yaw(points, yaw_deg):
    """
    [NEW] 對 LiDAR 座標系點雲進行偏航角 (Yaw) 旋轉
    KITTI LiDAR: x=前, y=左, z=上
    旋轉軸: z軸 (垂直) -> 修正路不是正對前方的情況
    """
    if yaw_deg == 0.0:
        return points

    theta = np.radians(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # 繞 Z 軸旋轉矩陣
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    # 注意：KITTI 座標系中 Y 是向左。
    # 旋轉公式：x' = x cos - y sin, y' = x sin + y cos
    # 若 yaw > 0 (逆時針)，則 y(左) 分量增加，x(前) 分量減少 -> 向左轉。符合直覺。
    
    x = points[:, 0]
    y = points[:, 1]
    
    points[:, 0] = x * c - y * s
    points[:, 1] = x * s + y * c
    
    return points

def apply_z_stretch(points, stretch_factor, ground_ref=-1.73):
    """
    [NEW] Z軸拉伸 (修正壓扁的車子)
    以地面(ground_ref)為基準點，只將上方的物體拉高。
    """
    if stretch_factor == 1.0: return points
    
    # 公式: 新高度 = (舊高度 - 地面) * 倍率 + 地面
    # 這樣地面 (-1.73) 的位置不會變，但比地面高的東西會變更高
    points[:, 2] = (points[:, 2] - ground_ref) * stretch_factor + ground_ref
    
    return points

def step3_pointcloud_to_voxel(points, pc_range, voxel_size, grid_size):
    """
    使用 VoxFormer 的 mapping 庫進行 Ray Casting (Ray Tracing)
    """
    # 1. 篩選範圍內的點 (這部分保留)
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) & \
           (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) & \
           (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    pts = points[keep]

    # mapping 模組預期輸入為 (N, 4)，包含 [x, y, z, intensity]
    # 如果目前的 points 只有 (N, 3)，我們需要補上一行 0 或 1
    if pts.shape[1] == 3:
        # 補上一列 0 (模擬 intensity)
        padding = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pts = np.hstack([pts, padding])
    
    # 2. 準備參數
    # LiDAR 原點 (Ray Casting 的發射源)
    origins = np.array([[0, 0, 0, 1]], dtype=np.float32) 
    
    # 3. 呼叫 mapping.compute_logodds_dp (核心邏輯)
    # 參數對應: points, sensor_origin, pc_range, range_indices, resolution
    # 注意: grid_size 在這裡是用 voxel_size 和 pc_range 推算出來的，mapping 內部會處理
    try:
        # compute_logodds_dp 會返回 log-odds 值
        visibility_map = mapping.compute_logodds_dp(
            pts, 
            origins[[0], :3], 
            pc_range, 
            range(pts.shape[0]), 
            voxel_size[0] # 假設 voxel 是正方體 (0.2)
        )
    except NameError:
        print("錯誤: 'mapping' 模組未載入，無法執行 Ray Casting。")
        return np.zeros(tuple(grid_size), dtype=np.float32)

    # 4. 轉換格式 (參考 lidar2voxel.py 的邏輯)
    # mapping 輸出的是一維陣列，需要 reshape
    # 原始輸出順序通常是 (Z, Y, X) 或 (Z, X, Y)，視編譯版本而定
    # 參考 lidar2voxel.py: reshape(-1, 32, 256, 256) -> swap axes -> (256, 256, 32)
    
    # 注意: lidar2voxel.py 裡面的 map_dims = [256, 256, 32]
    map_dims = grid_size # [256, 256, 32]
    
    visibility_map = np.asarray(visibility_map)
    # 根據 lidar2voxel.py 的 reshape 邏輯:
    visibility_map = visibility_map.reshape(-1, map_dims[2], map_dims[0], map_dims[1])
    # swapaxes(2,3) -> transpose(0,2,3,1) 
    visibility_map = np.swapaxes(visibility_map, 2, 3) 
    visibility_map = np.transpose(visibility_map, (0, 2, 3, 1))
    
    # 5. 二值化 (Binarize)
    # logodds > 0 為佔用 (Occupied), < 0 為空閒 (Free)
    # 這裡我們只取 Occupied = 1, Free = 0 (或者根據 VoxFormer 需求，Free 也要標記?)
    # 參考 lidar2voxel.py: 
    # recover[vis_occupy_indices] = 1
    # recover[vis_free_indices] = 0 (原本就是0)
    
    recover = np.zeros_like(visibility_map, dtype=np.uint8)
    vis_occupy_indices = (visibility_map > 0)
    recover[vis_occupy_indices] = 1
    
    # 注意：VoxFormer 訓練時可能需要 Free 空間的資訊 (Semantic Scene Completion)
    # 但 .pseudo 格式如果是用 pack() 壓縮的二進位，通常只存 0/1 狀態。
    # lidar2voxel.py 最終做的事： recover[...] = 1, 然後 pack(recover)
    
    # 回傳處理好的 voxel grid (尚未 pack)
    return recover

def process_single_image(model, idx, img_path, calib, args, directories):
    try:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. 深度估計
        depth = step1_predict_depth(model, img_path)

        print(f"[{base_name}] 深度範圍: {depth.min():.3f} ~ {depth.max():.3f} m")
        
        # 2. 決定 Scale
        # [NEW] 修改邏輯: 優先檢查手動高度
        if args.manual_height is not None:
            # 使用手動高度計算 Scale
            current_scale = KITTI_CAMERA_HEIGHT / args.manual_height
            print(f"[{base_name}] Manual Height: {args.manual_height:.2f}m -> Force Scale: {current_scale:.4f}")
        elif args.auto_scale:
            current_scale = estimate_scale_from_ground(depth, calib)
            print(f"[{base_name}] Auto Scale: {current_scale:.4f}")
        else:
            current_scale = args.fixed_scale

        if args.save_depth: np.save(os.path.join(directories['depth'], base_name+'.npy'), depth)
        if args.save_depth_vis: save_depth_vis(depth, os.path.join(directories['depth'], base_name+'.png'))

        # 3. 轉點雲 (帶入 Scale)
        points = step2_depth_to_pointcloud(depth, calib, args.max_depth, current_scale, args.focal_scale)

        # [NEW] 3-2. 手動俯仰角修正 (Pitch Correction)
        if args.manual_pitch != 0.0:
            print(f"[{base_name}] 手動俯仰角修正: {args.manual_pitch:.2f}°")
            points = apply_manual_pitch(points, args.manual_pitch)

        if args.manual_yaw != 0.0:
            print(f"[{base_name}] 手動偏航角修正: {args.manual_yaw:.2f}°")
            points = apply_manual_yaw(points, args.manual_yaw)

        if args.z_stretch != 1.0: 
            print(f"[{base_name}] 手動 Z-Stretch 修正: {args.z_stretch:.2f}x")
            points = apply_z_stretch(points, args.z_stretch)

        if points.shape[0] > 0:
            print(f"[{base_name}] LiDAR 共 {points.shape[0]} 個點雲")
            print(f"[{base_name}] LiDAR 點雲範圍(X,Y,Z): ")
            print(f"    X（前後）: {points[:,0].min():.3f} ~ {points[:,0].max():.3f} m")
            print(f"    Y（左右）: {points[:,1].min():.3f} ~ {points[:,1].max():.3f} m")
            print(f"    Z（高度）: {points[:,2].min():.3f} ~ {points[:,2].max():.3f} m")
        else:
            print(f"[{base_name}] LiDAR 點雲範圍: 無法計算")


        if args.save_lidar:
            bin_path = os.path.join(directories['lidar'], base_name+'.bin')
            np.hstack((points, np.zeros((len(points), 1)))).astype(np.float32).tofile(bin_path)

        # 4. 轉體素
        voxel = step3_pointcloud_to_voxel(points, PC_RANGE, VOXEL_SIZE, GRID_SIZE)
        occupied_count = np.sum(voxel > 0)
        print(f"[{base_name}] Voxel 體素資訊:")
        print(f"    Shape: {voxel.shape}")
        print(f"    占用體素數量: {occupied_count}")
        print(f"    總體素數量: {voxel.size}")
        print(f"    占用比例: {occupied_count / voxel.size:.2%}")
        print(f"    空閒體素數量: {voxel.size - occupied_count}")
        print(f"    空閒比例: {(voxel.size - occupied_count) / voxel.size:.2%}")
        print("-----------------------------------------------------")

        save_path = os.path.join(directories['voxel_packed' if args.compress else 'voxel'], 
                         base_name + ('.pseudo' if args.compress else '.npy'))

        if args.compress:
            # 確保 voxel 是 uint8 並且維度正確
            pack(voxel).tofile(save_path)
        else:
            np.save(save_path, voxel)
        
    except Exception as e:
        print(f"Error {img_path}: {e}")
        import traceback; traceback.print_exc()

def main():
    args = parse_args()
    calib = read_calib_file(args.calib_path)
    
    print(f"Loading Model: {args.model_type} ...")
    model = DepthAnything3.from_pretrained(args.model_type).to(torch.device("cuda"))
    
    # 準備檔案列表
    if args.image_dir:
        # 設定你想支援的所有副檔名 (建議包含大小寫，以免漏掉 .PNG 或 .JPG)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
        image_files = []
        
        # 針對每一個副檔名去抓檔案，然後加到清單中
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
            
        # 排序 (sorted) 並去重 (set)，避免如果有重複抓取的情況
        image_files = sorted(list(set(image_files)))
    else:
        image_files = [args.image_path]

    # 檢查是否為空
    if not image_files: 
        print(f"No images found in {args.image_dir}")
        return

    root_dir = args.save_dir if args.save_dir else os.path.dirname(args.image_dir.rstrip('/'))
    dirs = {
        'voxel': os.path.join(root_dir, 'pseudo_pc'),
        'voxel_packed': os.path.join(root_dir, 'pseudo_packed'),
        'depth': os.path.join(root_dir, 'depth_da3'),
        'lidar': os.path.join(root_dir, 'lidar_da3')
    }
    for k, v in dirs.items():
        if ((args.compress and k=='voxel_packed') or (not args.compress and k=='voxel') or
            (args.save_depth and k=='depth') or (args.save_lidar and k=='lidar')):
            os.makedirs(v, exist_ok=True)
            
    print(f"Processing {len(image_files)} images with Auto-Scale={args.auto_scale}...")
    
    for idx, img in enumerate(tqdm(image_files)):
        process_single_image(model, idx, img, calib, args, dirs)

if __name__ == "__main__":
    main()