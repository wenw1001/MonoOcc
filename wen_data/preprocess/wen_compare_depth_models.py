import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ================= 檔案路徑 =================
image_num = '000092'
img_path   = f'/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/image_2/{image_num}.png'
msnet_path = f'/mnt/data/mobilesterenet_result/depth/sequences/00/{image_num}.npy'
da3_path   = f'/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/depth_da3/{image_num}.npy'

# ================= 設定反轉邏輯 =================
# True 是 "近亮遠黑"，
# False，是 "近黑遠亮"
INVERT_MSNET = False 
INVERT_DA3   = False 
# ==============================================

def normalize_for_vis(data, invert=False, p_min=1, p_max=99):
    """
    將數據正規化，並根據需求反轉
    """
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 排除極端值
    vmin, vmax = np.percentile(data, [p_min, p_max])
    data_clipped = np.clip(data, vmin, vmax)
    
    # 正規化 0~1
    if vmax - vmin == 0:
        norm_data = data_clipped
    else:
        norm_data = (data_clipped - vmin) / (vmax - vmin)
    
    # 反轉
    if invert:
        norm_data = 1.0 - norm_data
        
    return norm_data

try:
    # 1. 讀取資料
    original_img = mpimg.imread(img_path)
    msnet_data = np.load(msnet_path).squeeze()
    da3_data = np.load(da3_path).squeeze()

    # 2. 處理深度圖
    vis_msnet = normalize_for_vis(msnet_data, invert=INVERT_MSNET)
    vis_da3   = normalize_for_vis(da3_data, invert=INVERT_DA3)

    # 3. 繪圖 (垂直排列：3列1行)
    # figsize=(寬, 高) -> 高度設大一點以便垂直排列
    plt.figure(figsize=(10, 18))

    # --- 第一張：原圖 (Top) ---
    plt.subplot(3, 1, 1)
    plt.imshow(original_img)
    plt.title(f"Original Image {image_num} (RGB)", fontsize=14)
    plt.axis('off')

    # --- 第二張：MSNet3D (Middle) ---
    plt.subplot(3, 1, 2)
    plt.imshow(vis_msnet, cmap='magma') # magma: 黑(低) -> 亮(高)
    plt.title(f"MSNet3D\n(Target: Near=Black, Far=Bright)", fontsize=14)
    plt.axis('off')
    # colorbar 放在右側
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # --- 第三張：Depth Anything 3 (Bottom) ---
    plt.subplot(3, 1, 3)
    plt.imshow(vis_da3, cmap='magma')
    plt.title(f"Depth Anything 3\n(Target: Near=Black, Far=Bright)", fontsize=14)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"發生錯誤: {e}")