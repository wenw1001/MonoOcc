import argparse
import torch
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, 'Depth-Anything-3', 'src')
from depth_anything_3.api import DepthAnything3

def test_single_image(image_path, model_type):
    result_path = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/da3_test_result"
    # 1. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在載入模型: {model_type} 到 {device}...")
    
    try:
        model = DepthAnything3.from_pretrained(model_type)
        model = model.to(device=device)
    except Exception as e:
        print(f"模型載入失敗: {e}")
        print("請確認你已安裝 depth_anything_3 並且網路連線正常(需下載權重)")
        return

    # 2. 讀取圖片
    if not os.path.exists(image_path):
        print(f"找不到圖片: {image_path}")
        return
    
    print(f"正在處理圖片: {image_path}")
    
    # DA3 的 inference 支援直接傳入路徑列表
    try:
        # 執行推論
        prediction = model.inference([image_path])
        
        # 獲取深度圖 (形狀為 [N, H, W])
        depth_map = prediction.depth[0]
        
        # 3. 分析數值 (重要！這決定了你是否需要調整 scale)
        print("-" * 30)
        print(f"深度圖形狀: {depth_map.shape}")
        print(f"資料型態: {depth_map.dtype}")
        print(f"最大值 (Max): {np.max(depth_map):.4f}")
        print(f"最小值 (Min): {np.min(depth_map):.4f}")
        print(f"平均值 (Mean): {np.mean(depth_map):.4f}")
        print("-" * 30)
        
        # 4. 視覺化並儲存
        filename = os.path.basename(image_path)
        save_name = os.path.join(result_path, f"vis_{filename}")
        print(f"正在儲存視覺化結果到: {save_name}")

        
        # 使用 matplotlib 畫出熱力圖
        plt.figure(figsize=(10, 5))
        plt.imshow(depth_map, cmap='Spectral_r') # 使用彩虹色階，紅色代表近，藍色代表遠
        plt.colorbar(label='Depth Value')
        plt.title(f'DA3 Depth Prediction: {filename}')
        plt.savefig(save_name)
        print(f"視覺化結果已儲存為: {save_name}")
        
        # 5. 儲存 npy 檔 (模擬 image2depth.sh 的輸出)
        npy_name = filename.replace('.png', '.npy').replace('.jpg', '.npy')
        npy_name = os.path.join(result_path, npy_name)
        np.save(npy_name, depth_map)
        print(f"深度數值已儲存為: {npy_name}")

    except Exception as e:
        print(f"推論過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 設定你的參數
    # 改成你實際的圖片路徑
    print("開始測試單張圖片的 Depth Anything 3 推論...")
    YOUR_IMAGE_PATH = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/image_2/000075.png" 
    
    # 模型型號
    MODEL_TYPE = "depth-anything/DA3NESTED-GIANT-LARGE"
    # MODEL_TYPE = "depth-anything/DA3-LARGE"
    
    test_single_image(YOUR_IMAGE_PATH, MODEL_TYPE)