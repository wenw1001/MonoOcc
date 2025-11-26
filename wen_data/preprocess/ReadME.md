# MonoOcc Preprocess Tools 使用說明

此目錄包含用於 MonoOcc 推論流程 **Stage 0 (前處理)** 的相關工具，主要用於將 2D 影像轉換為 3D 偽體素 (Pseudo Voxel)，並使用 **Depth Anything 3 (DA3)** 作為深度估計核心，取代舊有的 MobileStereoNet3D。

## 目錄結構

建議將腳本與資料整理如下：

```text
wen_data/
├── preprocess/
│   ├── preprocess_one_click.py  # [主程式] 批量/單張 自動化前處理工具
│   ├── wen_test_DA3.py          # [測試用] 單張影像深度數值檢查工具
│   └── README.md                # 本說明文件
│
└── wen_kitti/                   # 資料存放區範例
    └── 00/
        ├── calib.txt            # 相機參數
        ├── image_2/             # 原始影像
        └── pseudo_pc/           # [輸出] QPN 用的偽體素 (.npy)
```

## 工具 1：`preprocess_one_click.py` (主程式)

### 1. 簡介
這是前處理的核心工具。它將繁瑣的多步驟整合為「一鍵式」流程，直接生成 QPN 模型所需的輸入。

**主要功能：**
* **深度估計**：使用 DA3 推論高精度深度圖（並自動還原回原始解析度）。
* **點雲轉換**：利用相機內參將深度反向投影生成 3D 偽點雲。
* **體素化**：將點雲轉換為 QPN 模型所需的偽體素 (`pseudo_pc`) 格式。
* **批次處理**：支援處理整個資料夾的影像。
* **除錯功能**：可選擇性儲存中間產物（彩色深度圖、.bin 點雲檔）。

### 2. 環境需求
請確保在執行前啟動安裝了 `depth_anything_3` 的環境：
```bash
conda activate da3
```

### 3. 使用方法

請在 `MonoOcc` 專案根目錄下執行。

#### 基本用法（僅生成必要檔案）
這是最常用的模式，只會生成後續推論必要的 `.npy` 檔案。

```bash
python wen_data/preprocess/preprocess_one_click.py \
    --image-dir wen_data/wen_kitti/00/image_2 \
    --calib-path wen_data/wen_kitti/00/calib.txt \
    --save-dir wen_data/wen_kitti/00/pseudo_pc
```

#### 進階用法（儲存所有中間過程與視覺化）
如果你想檢查深度估計是否準確，或想用 Open3D 查看產生的點雲，請加上儲存參數。

```bash
python wen_data/preprocess/preprocess_one_click.py \
    --image-dir wen_data/wen_kitti/00/image_2 \
    --calib-path wen_data/wen_kitti/00/calib.txt \
    --save-dir wen_data/wen_kitti/00/pseudo_pc \
    --save-depth \
    --save-depth-vis \
    --save-lidar
```

### 4. 參數說明

| 參數 | 必填 | 說明 |
| :--- | :---: | :--- |
| `--image-dir` | 是* | 輸入影像的資料夾路徑。(*與 `--image-path` 擇一) |
| `--image-path` | 是* | 單張輸入影像的路徑。(*與 `--image-dir` 擇一) |
| `--calib-path` | **是** | KITTI 格式的 `calib.txt` 路徑 (用於讀取 P2 矩陣)。 |
| `--save-dir` | 否 | 偽體素 `.npy` 的輸出目錄。若不指定，預設為圖片目錄旁的 `pseudo_pc`。 |
| `--model-type` | 否 | DA3 模型版本 (預設: `depth-anything/DA3NESTED-GIANT-LARGE`)。 |
| `--max-depth` | 否 | 最大深度截斷距離 (預設: `80.0` 公尺)。 |
| `--save-depth` | 否 | [開關] 是否儲存原始深度數值 `.npy` (存於 `depth_da3/`)。 |
| `--save-depth-vis` | 否 | [開關] 是否儲存彩色深度熱力圖 `.png` (存於 `depth_da3/`)。 |
| `--save-lidar` | 否 | [開關] 是否儲存偽點雲 `.bin` (可用 Open3D 檢視，存於 `lidar_da3/`)。 |

## 工具 2：`wen_test_DA3.py` (測試用)

### 1. 簡介
這是一個輕量級的測試腳本，用於在執行大規模前處理之前，**快速驗證** DA3 模型是否能正常運作，並檢查輸出的**深度數值範圍**是否符合預期（公制深度 vs 相對深度）。

**功能：**
* **環境檢查**：確認 `depth_anything_3` 套件與 GPU 是否安裝正確。
* **數值分析**：列出深度圖的 Max, Min, Mean 值。
* **快速視覺化**：產生一張帶有色彩映射 (Color map) 的深度圖。

### 2. 使用方法

此腳本通常需要**手動修改**內部的圖片路徑變數。

1.  **編輯腳本**：打開 `wen_data/preprocess/wen_test_DA3.py`，找到並修改：
    ```python
    # 改成你實際想要測試的圖片路徑
    YOUR_IMAGE_PATH = "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/000000.png"
    ```

2.  **執行腳本**：
    ```bash
    python wen_data/preprocess/wen_test_DA3.py
    ```

### 3. 輸出解讀範例
觀察終端機輸出的 `Max` 數值：
* **Max > 10 且 < 200**：通常代表輸出為 **公制深度 (Metric Depth)**，單位為公尺（KITTI 場景通常在 80m 左右截斷）。**可以直接用於 `preprocess_one_click.py`。**
* **Max 接近 1.0 或 255**：代表是相對深度，可能需要額外的 Scale 參數校正。

---
## 常見問題 (FAQ)

* **Q: 執行時出現 `[WARN] Dependency gsplat is required...` 警告？**
    * **A:** 請忽略。這是 3D Gaussian Splatting 的依賴套件，我們只使用深度估計功能，不需要它。

* **Q: 輸出的深度圖大小與原圖不同？**
    * **A:** `preprocess_one_click.py` 已經內建自動 Resize 功能，推論後會強制將深度圖還原為原始影像的解析度，以確保點雲座標對齊，請放心使用。

* **Q: 程式執行後卡住沒有反應？**
    * **A:** 第一次執行時，程式需要下載並載入龐大的 `GIANT-LARGE` 模型到 GPU，這可能需要幾分鐘的時間，請耐心等待。