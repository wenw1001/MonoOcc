# -*- coding: utf-8 -*-
"""
這個腳本用於視覺化 MonoOcc 模型輸出的 3D 語義地圖，支援 .pkl 和 .bin 格式。
【新增功能】：
1. 支援按下 'C' 鍵手動截圖儲存。
2. 支援按下 'P' 鍵印出當前視角參數 (已修復 AttributeError)。
3. [本次新增] 支援 Command Line Arguments (執行參數)，可指定 image_file, base_dir, start, end。
"""
import sys
import os
import time
import argparse  # [新增] 用於解析參數
sys.path.insert(0, os.getcwd())

import numpy as np
import open3d as o3d
import pickle
import glob

# --- 1. 【使用者設定區】 ---

# True: 自動開啟視窗 -> 截圖 -> 關閉 -> 下一張 (無人值守模式)
# False: 手動模式 (需手動按 C 截圖，按 Q/ESC 關閉)
AUTO_MODE = True
# AUTO_MODE = False  

# 參數設定
MODE = 'file'
MAP_SHAPE = (256, 256, 32)
BIN_DTYPE = np.uint8 
UNPACK_BITS = True  

# ================= 資料夾與參數設定區 (修改後) =================

# [預設值] 如果執行時沒有輸入參數，會使用這裡的值
DEFAULT_IMAGE_FILE = "itri_campus_day"
# 預設的根目錄 (用於自動組建 base_dir)
DEFAULT_ROOT_DIR = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/'
DEFAULT_START = 0
DEFAULT_END = 400

# --- 解析命令列參數 ---
parser = argparse.ArgumentParser(description='Visual MonoOcc PKL results')
parser.add_argument('--seq', type=str, default=None, help='指定 image_file (例如: C330, 00, itri_campus)')
parser.add_argument('--dir', type=str, default=None, help='指定 base_dir (預測結果的完整資料夾路徑)')
parser.add_argument('--start', type=int, default=None, help='起始 Index')
parser.add_argument('--end', type=int, default=None, help='結束 Index (若跑到底則不需指定)')

args = parser.parse_args()

# [邏輯] 如果有輸入參數就用參數，否則用預設值
image_file = args.seq if args.seq is not None else DEFAULT_IMAGE_FILE
START_INDEX = args.start if args.start is not None else DEFAULT_START
END_INDEX = args.end if args.end is not None else DEFAULT_END

# [路徑邏輯]
if args.dir:
    # 如果使用者指定了完整路徑
    base_dir = args.dir
else:
    # 否則使用預設規則組建路徑
    base_dir = os.path.join(DEFAULT_ROOT_DIR, image_file, 'predictions_v1')

# ================= 動態生成 TEST_LIST =================
# 確保資料夾路徑結尾沒有多餘的斜線，避免 glob 出錯
if not base_dir.endswith('/'):
    base_dir += '/'

print(f"正在讀取資料夾: {base_dir}")
all_files = glob.glob(os.path.join(base_dir, "*.pkl"))
all_files = sorted(all_files)

# 處理切片範圍 (如果 END_INDEX 超過長度或為 None，Python slice 會自動處理)
target_files = all_files[START_INDEX : END_INDEX]
TEST_LIST = [os.path.splitext(os.path.basename(f))[0] for f in target_files]

print(f"資料夾內共有 {len(all_files)} 個檔案。")
print(f"已選取第 {START_INDEX} 到 {END_INDEX} 順位的檔案，共 {len(TEST_LIST)} 張。")
# print(f"範例 ID: {TEST_LIST[:3]} ...")

# ===============================================

def setting_info(image_num):
    """
    修改為使用全域變數 base_dir，以支援外部參數修改路徑
    """
    global MODE, INPUT_PATH, SAVE_FOLDER
    MODE = 'file'
    
    # 組合 Input Path (使用動態決定的 base_dir)
    INPUT_PATH = os.path.join(base_dir, f'{image_num}.pkl')
    
    # 組合 Save Folder (預設存在 predictions 同層的 3D_Map_Screenshots)
    # 使用 os.path.dirname(base_dir[:-1]) 可以回到上一層，或者直接存在 base_dir 下
    # 這裡保留原本邏輯：存在 predictions 資料夾旁邊的 '3D_Map_Screenshots'
    # 假設 base_dir 是 '.../predictions/'
    parent_dir = os.path.dirname(os.path.dirname(base_dir)) # 回到 image_file 層
    # 如果路徑結構比較複雜，最保險是直接存在 base_dir 裡面或旁邊，這裡設為 predictions 裡面
    SAVE_FOLDER = os.path.join(base_dir.rstrip('/'), '3D_Map_Screenshots')
    
    return MODE, INPUT_PATH, SAVE_FOLDER

# --- 2. 定義顏色對照表 ---
COLOR_MAP_RGB = [
    (0, 0, 0), (245, 150, 100), (245, 230, 100), (150, 60, 30), (180, 30, 80),
    (255, 0, 0), (30, 30, 255), (200, 40, 255), (90, 30, 150), (255, 0, 255),
    (255, 150, 255), (75, 0, 75), (75, 0, 175), (0, 200, 255), (50, 120, 255),
    (0, 175, 0), (0, 60, 135), (80, 240, 150), (150, 240, 255), (0, 0, 255)
]
COLOR_MAP = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in COLOR_MAP_RGB]
color_map_np = np.array(COLOR_MAP)
OCCUPIED_COLOR = (1.0, 0.0, 0.0)

def load_and_process_file(file_path):
    try:
        # print(f"處理檔案: {file_path}") # 減少 log
        file_ext = os.path.splitext(file_path)[1].lower()
        output_voxel_flat = None
        is_packed_bits = False

        if file_ext == '.pkl':
            with open(file_path, 'rb') as f:
                out_dict = pickle.load(f)
            if isinstance(out_dict, dict) and 'output_voxel' in out_dict:
                output_voxel_flat = out_dict['output_voxel']
            else:
                pass
        elif file_ext == '.bin':
            output_voxel_flat = np.fromfile(file_path, dtype=BIN_DTYPE)
            if UNPACK_BITS:
                output_voxel_flat = np.unpackbits(output_voxel_flat)
                is_packed_bits = True

        if output_voxel_flat is None: return None, None
        if output_voxel_flat.size != np.prod(MAP_SHAPE): return None, None

        semantic_map_np = output_voxel_flat.reshape(MAP_SHAPE)
        occupied_indices = np.where(semantic_map_np > 0)
        if occupied_indices[0].size == 0: return None, None

        points = np.stack(occupied_indices, axis=1)
        if is_packed_bits:
            colors = np.tile(OCCUPIED_COLOR, (points.shape[0], 1))
        else:
            class_ids = semantic_map_np[occupied_indices].astype(int)
            class_ids = np.clip(class_ids, 0, len(color_map_np) - 1)
            colors = color_map_np[class_ids]
        return points, colors
    except Exception as e:
        print(f"錯誤: {e}")
        return None, None

def main(image_num):
    print(f"[{'AUTO' if AUTO_MODE else 'MANUAL'}] 正在處理: {image_num} ...")
    MODE, INPUT_PATH, SAVE_FOLDER = setting_info(image_num)
    
    files_to_process = []
    if MODE == 'file':
        if os.path.isfile(INPUT_PATH): files_to_process.append(INPUT_PATH)
    elif MODE == 'folder':
        if os.path.isdir(INPUT_PATH):
            files_to_process = sorted(glob.glob(os.path.join(INPUT_PATH, '*.pkl')) + 
                                      glob.glob(os.path.join(INPUT_PATH, '*.bin')))

    if not files_to_process:
        print(f"沒有找到檔案: {INPUT_PATH}")
        return

    all_points, all_colors = [], []
    for fp in files_to_process:
        p, c = load_and_process_file(fp)
        if p is not None:
            all_points.append(p)
            all_colors.append(c)

    if not all_points:
        print("無法產生點雲。")
        return

    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    # 這裡註解掉說明，避免自動模式一直洗版
    # print("-" * 50)
    # print("【操作說明】...")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"MonoOcc Viz {image_num}", width=1920, height=1080)

    def capture_action(vis):
        """ 執行截圖動作的核心邏輯 """
        global t
        filename = f"{image_num}_前.png"
        full_path = os.path.join(SAVE_FOLDER, filename)
        vis.capture_screen_image(full_path)
        return False
    
    def save_snapshot(vis):
        """ 手動按 C 觸發 """
        global t
        if t%2 == 0: 
            filename = f"{image_num}_前.png"
        else: 
            filename = f"{image_num}_上.png"
            t+=1
        full_path = os.path.join(SAVE_FOLDER, filename)
        vis.capture_screen_image(full_path)
        print(f" >> [截圖成功] {filename}")
        return False

    def print_view_info(vis):
        """ 手動按 P 觸發 """
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic 
        inv_ext = np.linalg.inv(extrinsic)
        camera_pos = inv_ext[:3, 3]
        front_vec = -inv_ext[:3, 2] 
        up_vec = inv_ext[:3, 1]     
        lookat_point = camera_pos + front_vec * 100.0 
        
        print("\n# === 請複製以下代碼到您的 main 函式中 (取代原本的視角設定) ===")
        print(f"ctr = vis.get_view_control()")
        print(f"ctr.set_lookat([{lookat_point[0]:.2f}, {lookat_point[1]:.2f}, {lookat_point[2]:.2f}])")
        print(f"ctr.set_front([{front_vec[0]:.2f}, {front_vec[1]:.2f}, {front_vec[2]:.2f}])")
        print(f"ctr.set_up([{up_vec[0]:.2f}, {up_vec[1]:.2f}, {up_vec[2]:.2f}])")
        print(f"ctr.set_zoom(0.5) # Zoom 無法精確反推，請手動微調此數值")
        print("# ==========================================================\n")
        return False
    
    def auto_capture_and_close(vis):
        """ 自動模式的回調函式 """
        capture_action(vis)
        vis.destroy_window() 
        return False

    global t
    t = 0
    vis.register_key_callback(ord('C'), save_snapshot)
    vis.register_key_callback(ord('P'), print_view_info) 
    
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    
    # 預設視角
    ctr = vis.get_view_control()
    ctr.set_lookat([128, 128, 16]) 
    ctr.set_front([-0.91, -0.03, 0.42])
    ctr.set_up([0.42, -0.01, 0.91])
    ctr.set_zoom(0.7)

    if AUTO_MODE:
        vis.register_animation_callback(auto_capture_and_close)

    vis.run()

if __name__ == "__main__":
    total = len(TEST_LIST)
    for idx, i in enumerate(TEST_LIST):
        if AUTO_MODE:
             # 簡單的進度條
             print(f"進度: {idx+1}/{total} (ID: {i})", end='\r')
        main(i)