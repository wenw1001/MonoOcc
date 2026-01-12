"""
將 MonoOcc 的 3D Map 截圖與原始 RGB 圖垂直拼接，並輸出成影片。
"""
import cv2
import os
import glob
import numpy as np
import argparse

# ================= 預設設定 (Defaults) =================
# 如果沒有輸入參數，就會使用這邊的數值
DEFAULT_RGB_FOLDER = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day_test_kitti/image_2'
DEFAULT_RESULT_FOLDER = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/itri_campus_day_test_kitti/predictions_v8/3D_Map_Screenshots'
DEFAULT_OUTPUT_NAME = 'monoocc.mp4'
DEFAULT_FPS = 5.0  # 20.0
DEFAULT_FILTER = "_前"

# 裁剪預設值 (Top, Bottom, Left, Right)

DEFAULT_CROP = [330, 150, 339, 340] # ITRI campus
DEFAULT_CROP_KITTI = [400, 304, 339, 340] # KITTI
# =======================================================

def get_args():
    parser = argparse.ArgumentParser(description="將 MonoOcc 的 3D Map 截圖與原始 RGB 圖垂直拼接並輸出影片。")

    # 路徑參數
    parser.add_argument('--rgb', type=str, default=DEFAULT_RGB_FOLDER, help='原始 RGB 圖片資料夾路徑')
    parser.add_argument('--result', type=str, default=DEFAULT_RESULT_FOLDER, help='3D Map 截圖資料夾路徑')
    parser.add_argument('--output', type=str, default=None, help='輸出影片的完整路徑 (選填，若不填則預設存於 result 資料夾)')
    
    # 影片參數
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help='影片 FPS')
    parser.add_argument('--filter', type=str, default=DEFAULT_FILTER, help='檔名過濾關鍵字 (例如: _前)')

    # 裁剪參數
    parser.add_argument('--top', type=int, default=DEFAULT_CROP[0], help='上方裁剪像素')
    parser.add_argument('--bottom', type=int, default=DEFAULT_CROP[1], help='下方裁剪像素')
    parser.add_argument('--left', type=int, default=DEFAULT_CROP[2], help='左方裁剪像素')
    parser.add_argument('--right', type=int, default=DEFAULT_CROP[3], help='右方裁剪像素')

    return parser.parse_args()

def resize_to_width(img, target_width):
    """將圖片等比例縮放到指定寬度"""
    h, w = img.shape[:2]
    # 如果原始圖寬度已經跟目標一樣，就不用縮放 (避免失真)
    if w == target_width:
        return img
        
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

def crop_image(img, t, b, l, r):
    """ 對圖片進行裁剪 """
    h, w = img.shape[:2]
    
    if t + b >= h or l + r >= w:
        print(f"警告: 裁剪範圍過大 (H:{h}, W:{w}) vs (Cut_H:{t+b}, Cut_W:{l+r})，跳過裁剪。")
        return img
    
    end_h = h - b if b > 0 else h
    end_w = w - r if r > 0 else w
    
    # 防呆：確保座標合理
    end_h = max(t, end_h)
    end_w = max(l, end_w)
    
    cropped = img[t:end_h, l:end_w]
    return cropped

def get_image_id(filename):
    name_no_ext = os.path.splitext(filename)[0]
    image_id = name_no_ext.split('_')[0]
    return image_id

def main():
    args = get_args()

    # 設定輸出路徑
    if args.output is None:
        output_video_path = os.path.join(args.result, DEFAULT_OUTPUT_NAME)
    else:
        output_video_path = args.output

    # 搜尋檔案
    search_pattern = os.path.join(args.result, f"*{args.filter}*.png")
    result_files = sorted(glob.glob(search_pattern))

    if not result_files:
        print(f"錯誤: 在 '{args.result}' 找不到符合 '{args.filter}' 的檔案")
        return

    print("="*50)
    print(f"RGB 路徑   : {args.rgb}")
    print(f"截圖路徑   : {args.result}")
    print(f"輸出影片   : {output_video_path}")
    print(f"找到張數   : {len(result_files)} 張")
    print(f"裁切設定   : Top:{args.top}, Bottom:{args.bottom}, Left:{args.left}, Right:{args.right}")
    print("="*50)
    
    video_writer = None
    
    for i, res_path in enumerate(result_files):
        # 1. 讀取結果圖
        img_result = cv2.imread(res_path)
        if img_result is None: 
            print(f"無法讀取: {res_path}")
            continue

        # --- [裁剪] ---
        img_result_cropped = crop_image(img_result, args.top, args.bottom, args.left, args.right)
        
        # 2. 讀取原始 RGB 圖
        filename = os.path.basename(res_path)
        img_id = get_image_id(filename)
        rgb_path = os.path.join(args.rgb, f"{img_id}.png")
        
        if not os.path.exists(rgb_path):
             rgb_path = os.path.join(args.rgb, f"{img_id}.jpg")

        img_rgb = cv2.imread(rgb_path)
        if img_rgb is None: 
            print(f"找不到對應 RGB: {rgb_path}")
            continue

        # 3. 調整尺寸 (強制將原圖縮放至與結果圖同寬)
        target_width = img_result_cropped.shape[1]
        img_rgb_resized = resize_to_width(img_rgb, target_width)

        # 4. 垂直拼接
        combined_frame = np.vstack((img_rgb_resized, img_result_cropped))

        # 5. 寫入影片
        if video_writer is None:
            h, w = combined_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, args.fps, (w, h))
            print(f"開始寫入影片，解析度: {w}x{h}, FPS: {args.fps}")

        video_writer.write(combined_frame)
        
        if (i + 1) % 10 == 0:
            print(f"進度: {i + 1}/{len(result_files)}")

    if video_writer:
        video_writer.release()
        print(f"\n完成！影片已儲存: {output_video_path}")
    else:
        print("未生成影片 (可能沒有成功讀取圖片)。")

if __name__ == "__main__":
    main()