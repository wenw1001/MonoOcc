"""
conda activate da3

python /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click.py \
    --image-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/image_2 \
    --calib-path /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/calib.txt \
    --save-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00 \ # 預設在影像的前一個資料夾建立輸出資料夾
    --save-depth \
    --save-depth-vis \
    --save-lidar \
    --compress

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

# 嘗試匯入 Depth Anything 3
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("錯誤: 找不到 'depth_anything_3'。請確認你已啟動安裝了 DA3 的環境")
    sys.exit(1)

# --- 預設參數 ---
DEFAULT_CALIB_K = np.array([
    [721.5377, 0.0, 609.5593],
    [0.0, 721.5377, 172.8540],
    [0.0, 0.0, 1.0]
])

PC_RANGE = np.array([0, -25.6, -2.0, 51.2, 25.6, 4.4])
VOXEL_SIZE = np.array([0.2, 0.2, 0.2])
GRID_SIZE = np.array([256, 256, 32])

def parse_args():
    parser = argparse.ArgumentParser(description='One-click preprocess: Image -> Depth -> Lidar -> Voxel')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image-path', type=str, help='Path to a single image file')
    group.add_argument('--image-dir', type=str, help='Path to a directory containing images')
    
    parser.add_argument('--calib-path', type=str, required=True, help='Path to KITTI calib.txt')
    # 修改說明：save-dir 現在是用於指定"根"輸出目錄，或者你可以不指定讓它自動產生
    parser.add_argument('--save-dir', type=str, help='Custom output directory. If not set, defaults to sibling directories (pseudo_pc / pseudo_packed)')
    
    parser.add_argument('--model-type', type=str, default='depth-anything/DA3NESTED-GIANT-LARGE', help='DA3 Model ID')
    parser.add_argument('--max-depth', type=float, default=80.0, help='Max depth to keep (meters)')

    parser.add_argument('--compress', action='store_true', help='Save as compressed .pseudo format (uint8 bit-packed)')
    parser.add_argument('--save-depth', action='store_true', help='Save raw depth map (.npy)')
    parser.add_argument('--save-depth-vis', action='store_true', help='Save colored depth map visualization (.png)')
    parser.add_argument('--save-lidar', action='store_true', help='Save pseudo point cloud (.bin)')
    
    return parser.parse_args()

def pack(array):
    """位元壓縮: 8 boolean -> 1 uint8"""
    array = array.reshape((-1)).astype(np.uint8)
    compressed = (array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | 
                  array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8])
    return np.array(compressed, dtype=np.uint8)

def read_calib_file(filepath):
    if not os.path.exists(filepath): return DEFAULT_CALIB_K
    k = None
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line.startswith('P2:'):
                values = [float(x) for x in line.split()[1:]]
                p2 = np.array(values).reshape(3, 4)
                k = p2[:3, :3]
                break
    return k if k is not None else DEFAULT_CALIB_K

def step1_predict_depth(model, image_path):
    import cv2
    original_img = cv2.imread(image_path)
    orig_h, orig_w = original_img.shape[:2]
    
    pred = model.inference([image_path])
    depth_map = pred.depth[0]
    
    if depth_map.shape[0] != orig_h or depth_map.shape[1] != orig_w:
        depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
    return depth_map

def save_depth_vis(depth_map, save_path):
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(save_path, depth_vis)

def step2_depth_to_pointcloud(depth_map, K, max_depth=80.0):
    rows, cols = depth_map.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    mask = (depth_map > 0) & (depth_map < max_depth)
    depth = depth_map[mask]
    u = c[mask]
    v = r[mask]
    cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth
    x_l = z_c
    y_l = -x_c
    z_l = -y_c
    return np.stack([x_l, y_l, z_l], axis=-1)

def step3_pointcloud_to_voxel(points, pc_range, voxel_size, grid_size):
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) & \
           (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) & \
           (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    points = points[keep]
    if points.shape[0] == 0: return np.zeros(tuple(grid_size), dtype=np.float32)
    coords = (points - pc_range[:3]) / voxel_size
    coords = np.floor(coords).astype(np.int32)
    unique_coords = np.unique(coords, axis=0)
    voxel_grid = np.zeros(tuple(grid_size), dtype=np.float32)
    x, y, z = unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]
    valid_mask = (x >= 0) & (x < grid_size[0]) & (y >= 0) & (y < grid_size[1]) & (z >= 0) & (z < grid_size[2])
    voxel_grid[x[valid_mask], y[valid_mask], z[valid_mask]] = 1.0
    return voxel_grid

def process_single_image(model, img_path, K, args, directories):
    try:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]

        # 1. Image -> Depth
        depth_map = step1_predict_depth(model, img_path)
        
        # Save Depth
        if args.save_depth:
            np.save(os.path.join(directories['depth'], base_name + '.npy'), depth_map)
        if args.save_depth_vis:
            save_depth_vis(depth_map, os.path.join(directories['depth'], base_name + '.png'))

        # 2. Depth -> LiDAR
        points = step2_depth_to_pointcloud(depth_map, K, args.max_depth)
        
        # Save LiDAR
        if args.save_lidar:
            points_with_intensity = np.hstack((points, np.zeros((points.shape[0], 1))))
            points_with_intensity.astype(np.float32).tofile(os.path.join(directories['lidar'], base_name + '.bin'))

        # 3. Points -> Voxel
        voxel_grid = step3_pointcloud_to_voxel(points, PC_RANGE, VOXEL_SIZE, GRID_SIZE)
        
        # 4. Save Final Voxel (分開資料夾)
        if args.compress:
            # 存到 pseudo_packed 資料夾
            save_path = os.path.join(directories['voxel_packed'], base_name + '.pseudo')
            packed_voxel = pack(voxel_grid)
            packed_voxel.tofile(save_path)
        else:
            # 存到 pseudo_pc 資料夾
            save_path = os.path.join(directories['voxel'], base_name + '.npy')
            np.save(save_path, voxel_grid)
        
    except Exception as e:
        print(f"處理 {img_path} 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()
    K = read_calib_file(args.calib_path)
    
    print(f"正在載入模型: {args.model_type} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model_type).to(device)
    
    image_files = []
    if args.image_path:
        image_files.append(args.image_path)
        base_out_dir = os.path.dirname(args.image_path)
    else:
        search_pattern = os.path.join(args.image_dir, "*.png")
        image_files = sorted(glob.glob(search_pattern))
        base_out_dir = os.path.dirname(args.image_dir.rstrip('/'))

    if not image_files:
        print("錯誤: 找不到任何圖片檔案。")
        return

    # --- 資料夾路徑設定邏輯 ---
    # 如果使用者指定了 save-dir，我們就在那個目錄下建立子資料夾
    # 如果沒指定，就在圖片目錄上一層建立子資料夾
    root_save_dir = args.save_dir if args.save_dir else base_out_dir
    
    directories = {
        'voxel': os.path.join(root_save_dir, 'pseudo_pc'),       # 存放 .npy
        'voxel_packed': os.path.join(root_save_dir, 'pseudo_packed'), # 存放 .pseudo
        'depth': os.path.join(root_save_dir, 'depth_da3'),
        'lidar': os.path.join(root_save_dir, 'lidar_da3')
    }

    # 建立需要的資料夾
    if args.compress:
        if not os.path.exists(directories['voxel_packed']): os.makedirs(directories['voxel_packed'])
    else:
        if not os.path.exists(directories['voxel']): os.makedirs(directories['voxel'])
        
    if args.save_depth or args.save_depth_vis:
        if not os.path.exists(directories['depth']): os.makedirs(directories['depth'])
    if args.save_lidar:
        if not os.path.exists(directories['lidar']): os.makedirs(directories['lidar'])
    
    print(f"準備處理 {len(image_files)} 張圖片...")
    if args.compress:
        print(f"Voxel 輸出 (壓縮 .pseudo): {directories['voxel_packed']}")
    else:
        print(f"Voxel 輸出 (原始 .npy):   {directories['voxel']}")

    iterator = tqdm(image_files, desc="Processing") if 'tqdm' in sys.modules else image_files
    for img_path in iterator:
        process_single_image(model, img_path, K, args, directories)
        
    print("所有處理完成！")

if __name__ == "__main__":
    main()