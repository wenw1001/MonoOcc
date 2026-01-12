import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# ================= 參數設定區 =================

# 1. 路徑設定
INPUT_IMG_DIR   = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day/image_2'
INPUT_CALIB_PATH = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day/calib.txt'
OUTPUT_BASE_DIR = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day/'

# 2. 輸出資料夾名稱
FOLDER_NAME_KITTI  = 'image_2_kitti'   # 存放 KITTI 大小的資料夾
FOLDER_NAME_CUSTOM = 'image_2_custom'  # 存放自訂裁切的資料夾
OUTPUT_CALIB_NAME  = 'calib_kitti.txt' # 新的 calibration 檔名

# 3. [設定] KITTI 目標尺寸 (寬, 高)
KITTI_TARGET_SIZE = (1241, 376) 

# 4. [設定] 自訂裁切保留比例 (0.74 = 保留上面 74%)
KEEP_RATIO = 0.74

# ============================================

def process_kitti_style(img, target_w, target_h):
    """
    影像處理: Resize -> Center Crop
    回傳: (處理後的圖片, 縮放比例 scale, 上方裁切量 crop_top)
    """
    h, w = img.shape[:2]
    
    # 1. 計算縮放比例 (以寬度為主)
    scale = target_w / w
    new_w = target_w
    new_h = int(h * scale)
    
    # 縮放影像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. 高度裁切 (Center Crop)
    crop_top = 0
    if new_h > target_h:
        crop_top = (new_h - target_h) // 2
        final_img = resized_img[crop_top : crop_top + target_h, :]
    else:
        final_img = resized_img
        
    return final_img, scale, crop_top

def process_custom_ratio(img, ratio):
    h = img.shape[0]
    cut_point = int(h * ratio)
    return img[0:cut_point, :]

def update_calib_content(calib_path, scale, crop_top):
    """
    讀取原始 calib，計算並回傳新的 calib 字串內容
    """
    new_lines = []
    
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        
    print(f"正在重新計算 Calibration (Scale: {scale:.5f}, Crop Top: {crop_top})...")
    
    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue
            
        key = parts[0] # e.g., "P2:"
        
        # 我們主要修正 P0, P1, P2, P3 這些投影矩陣
        if key.startswith('P'):
            # 解析原本的數值 (12個 float)
            vals = np.array([float(x) for x in parts[1:]]).reshape(3, 4)
            
            # --- 修正邏輯 ---
            # P 矩陣 = [fx  0 cx tx]
            #          [ 0 fy cy ty]
            #          [ 0  0  1  0]
            
            # 1. 縮放: 第一列(x) 和 第二列(y) 都要乘上 scale
            #    注意: tx, ty 也會跟著變 (因為 fx*baseline)，所以整列乘 scale 是對的
            vals[0, :] *= scale
            vals[1, :] *= scale
            
            # 2. 裁切: 只有 cy (光學中心 y) 需要減去 crop_top
            #    cy 在矩陣的 index 是 [1, 2]
            vals[1, 2] -= crop_top
            
            # 3. 轉回字串
            vals_flat = vals.flatten()
            new_vals_str = " ".join([f"{v:.5f}" for v in vals_flat])
            new_lines.append(f"{key} {new_vals_str}\n")
            
        else:
            # R0_rect, Tr 等不需要變動，直接複製
            new_lines.append(line)
            
    return "".join(new_lines)

def main():
    # 檢查輸入
    if not os.path.exists(INPUT_IMG_DIR):
        print(f"錯誤：找不到影像資料夾 {INPUT_IMG_DIR}")
        return
    if not os.path.exists(INPUT_CALIB_PATH):
        print(f"錯誤：找不到校正檔 {INPUT_CALIB_PATH}")
        return

    # 建立輸出資料夾
    path_kitti = os.path.join(OUTPUT_BASE_DIR, FOLDER_NAME_KITTI)
    path_custom = os.path.join(OUTPUT_BASE_DIR, FOLDER_NAME_CUSTOM)
    os.makedirs(path_kitti, exist_ok=True)
    os.makedirs(path_custom, exist_ok=True)
    
    # 搜尋圖片
    image_files = sorted(glob.glob(os.path.join(INPUT_IMG_DIR, "*.png")) + 
                         glob.glob(os.path.join(INPUT_IMG_DIR, "*.jpg")))
    
    if len(image_files) == 0:
        print("找不到任何圖片。")
        return

    print(f"找到 {len(image_files)} 張圖片，開始處理...")

    # --- 步驟 1: 處理第一張圖以取得縮放參數 ---
    # 我們只需要算一次 scale 和 crop_top，因為所有圖片尺寸應該都一樣
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print("無法讀取第一張圖片，程式終止。")
        return
        
    # 試算參數
    _, scale_factor, crop_top_pixels = process_kitti_style(first_img, KITTI_TARGET_SIZE[0], KITTI_TARGET_SIZE[1])
    
    # --- 步驟 2: 產生並儲存新的 Calib 檔 ---
    new_calib_content = update_calib_content(INPUT_CALIB_PATH, scale_factor, crop_top_pixels)
    output_calib_path = os.path.join(OUTPUT_BASE_DIR, OUTPUT_CALIB_NAME)
    
    with open(output_calib_path, 'w') as f:
        f.write(new_calib_content)
    print(f"✅ 新的校正檔已儲存: {output_calib_path}")

    # --- 步驟 3: 批次處理所有圖片 ---
    for file_path in tqdm(image_files):
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path)
        if img is None: continue
            
        # A. KITTI Style (Resize + Crop)
        img_kitti, _, _ = process_kitti_style(img, KITTI_TARGET_SIZE[0], KITTI_TARGET_SIZE[1])
        cv2.imwrite(os.path.join(path_kitti, filename), img_kitti)
        
        # B. Custom Ratio (Top 77%)
        img_custom = process_custom_ratio(img, KEEP_RATIO)
        cv2.imwrite(os.path.join(path_custom, filename), img_custom)

    print("\n全部完成！")

if __name__ == "__main__":
    main()