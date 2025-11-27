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
        ├── image_2/             # 原始輸入影像
        ├── depth_da3/           # [輸出，可選] 原始深度資訊（.npy）和深度圖視覺化 (.png)
        ├── lidar_da3/           # [輸出，可選] 偽點雲 (.bin)
        ├── pseudo_pc/           # [標準輸出] .npy 格式 (8.4MB/檔)
        └── pseudo_packed/       # [壓縮輸出] .pseudo 格式 (262KB/檔)，QPN 用的偽體素預設為此
```

## 工具 1：`preprocess_one_click.py` (主程式)

### 1. 簡介
這是前處理的核心工具。它將繁瑣的多步驟整合為「一鍵式」流程，直接生成 QPN 模型所需的輸入。

**主要功能：**
* **深度估計**：使用 DA3 推論高精度深度圖（並自動還原回原始解析度）。
* **點雲轉換**：利用相機內參將深度反向投影生成 3D 偽點雲。
* **體素化**：將點雲轉換為 QPN 模型所需的偽體素 (`pseudo_pc`) 格式。
* **空間節省**：支援 `.pseudo` 位元壓縮格式，大幅減少儲存空間 (32x 壓縮率)。
* **自動分流**：自動將 `.npy` 與 `.pseudo` 存放在不同資料夾，避免混淆。
* **批次處理**：支援處理整個資料夾的影像。
* **除錯功能**：可選擇性儲存中間產物（彩色深度圖、.bin 點雲檔）。

### 2. 環境需求
請確保在執行前啟動安裝了 `depth_anything_3` 的環境：
```bash
conda activate da3
```
若尚未安裝，可參照以下指令安裝：
```bash
conda create -n da3 python=3.10
conda activate da3
cd MonoOcc
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything3

# 安裝依賴
pip install xformers "torch>=2" torchvision
# 安裝 Depth Anything 3
pip install -e .
```

### 3. 使用方法

請在 `MonoOcc` 專案根目錄下執行。

#### 模式 A：標準輸出 (推薦用於除錯/小量資料)
產生標準的 `.npy` (Float32) 檔案，檔案較大 (~8.4MB)，但讀取方便。檔案將存於 `pseudo_pc/` 目錄。

```bash
python wen_data/preprocess/preprocess_one_click.py \
    --image-dir wen_data/wen_kitti/00/image_2 \
    --calib-path wen_data/wen_kitti/00/calib.txt \
    --save-dir wen_data/wen_kitti/00
```

#### 模式 B：壓縮輸出 (推薦用於大量資料)
加入 `--compress` 參數。產生位元壓縮的 `.pseudo` (uint8) 檔案，檔案極小 (~262KB)。檔案將自動存於 `pseudo_packed/` 目錄。

```bash
python wen_data/preprocess/preprocess_one_click.py \
    --image-dir wen_data/wen_kitti/00/image_2 \
    --calib-path wen_data/wen_kitti/00/calib.txt \
    --save-dir wen_data/wen_kitti/00 \
    --compress
```

#### 進階用法（儲存所有中間過程）
如果你想檢查深度估計是否準確，或想用 Open3D 查看產生的點雲，請加上儲存參數。

```bash
python wen_data/preprocess/preprocess_one_click.py \
    --image-dir wen_data/wen_kitti/00/image_2 \
    --calib-path wen_data/wen_kitti/00/calib.txt \
    --save-dir wen_data/wen_kitti/00 \
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
| `--save-dir` | 否 | 偽體素 `.npy`/`.pseudo` 輸出的**根目錄**。腳本會在其下自動建立 `pseudo_pc` 或 `pseudo_packed` 子資料夾。 若不指定，預設為與圖片目錄同一層。 |
| `--compress` | **否** | 若啟用，將輸出壓縮的 `.pseudo` 格式並存入 `pseudo_packed/`。若未啟用，則輸出 `.npy` 至 `pseudo_pc/`。 |
| `--model-type` | 否 | DA3 模型版本 (預設: `depth-anything/DA3NESTED-GIANT-LARGE`)。 |
| `--max-depth` | 否 | 最大深度截斷距離 (預設: `80.0` 公尺)。 |
| `--save-depth` | 否 | [開關] 是否儲存原始深度數值 `.npy` (存於 `depth_da3/`)。 |
| `--save-depth-vis` | 否 | [開關] 是否儲存彩色深度熱力圖 `.png` (存於 `depth_da3/`)。 |
| `--save-lidar` | 否 | [開關] 是否儲存偽點雲 `.bin` (可用 Open3D 檢視，存於 `lidar_da3/`)。 |

> 補充說明：關於 `--max-depth` 參數
> **`--max-depth` (預設值: 80.0 公尺)** 是一個關鍵的過濾參數，用於決定從 2D 深度圖轉換為 3D 點雲時的**最遠有效距離**。
> **為什麼需要設定這個限制？**
> 1.  **去除「天空」與「無限遠」雜訊**：
    Depth Anything 3 (DA3) 的推論能力非常強，甚至能估計出天空、極遠處建築物或背景的深度（可能達數百公尺）。在轉換為 3D 偽點雲時，這些極遠的點會形成無意義的背景雜訊，干擾模型對車輛周遭場景的判斷。
> 2.  **對齊 KITTI/LiDAR 規格**：
    KITTI 資料集所使用的 LiDAR 感測器有效偵測範圍約為 **80 公尺**。為了讓生成的偽點雲 (Pseudo LiDAR) 的分佈特性與真實 LiDAR 數據（Ground Truth）保持一致，通常會截斷超過此距離的點。
> 3.  **優化體素化 (Voxelization) 效果**：
    MonoOcc 模型的關注範圍（Voxel Grid）通常集中在車前 **51.2 公尺** 以內。保留過遠的點（如 150m 外的點）不僅對語意佔用預測沒有幫助，還會增加檔案大小並可能影響座標轉換的正規化過程。
> **建議**：一般情況下請保持預設值 `80.0`。除非您的應用場景需要偵測極遠處的物體，否則不建議調大此數值。

### 5. 可選用的 DA3 模型列表與規格

`preprocess_one_click.py` 預設使用最強大的 `depth-anything/DA3NESTED-GIANT-LARGE` 模型。您可以透過 `--model-type` 參數切換其他模型，以平衡效能與速度。

| 模型名稱 (參數值) | 參數量 | 功能特性 (Features) | 授權 (License) |
| :--- | :--- | :--- | :--- |
| **[depth-anything/DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)** | **1.40B** | **(預設, 最強)** 支援公制深度 (Metric Depth)、相對深度、姿態估計、姿態條件、3DGS、天空分割。 | CC BY-NC 4.0 |
| [depth-anything/DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT) | 1.15B | 支援相對深度、姿態估計、姿態條件、3DGS。 | CC BY-NC 4.0 |
| [depth-anything/DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE) | 0.35B | 支援相對深度、姿態估計、3DGS。 | CC BY-NC 4.0 |
| [depth-anything/DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | 0.12B | 支援相對深度、姿態估計、3DGS。速度較快。 | Apache 2.0 |
| [depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | 0.08B | 支援相對深度、姿態估計、3DGS。最輕量化。 | Apache 2.0 |
| [depth-anything/DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | 0.35B | 專注於 **公制深度 (Metric Depth)** 與天空分割。 | Apache 2.0 |
| [depth-anything/DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE) | 0.35B | 專注於相對深度與天空分割。 | Apache 2.0 |

> **功能圖例說明**：
> * **公制深度 (Met. Depth)**：輸出真實世界的距離單位（如公尺）。
> * **相對深度 (Rel. Depth)**：輸出物體間的遠近關係，無絕對單位。
> * **3DGS (GS)**：支援 3D Gaussian Splatting 渲染。

注意：
* VRAM 需求：Giant 系列模型 (1B+) 需要較大的顯卡記憶體 (建議 24GB+)。若遇到 OOM (Out of Memory) 錯誤，請嘗試切換至 Large 或 Base 版本。
* 授權：請留意部分模型 (Giant/Large) 採用 CC BY-NC 4.0 (僅限非商業用途)，而 Base/Small/Metric 則採用 Apache 2.0。

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

## 工具 3：`wen_compare_depth_models.py` (深度模型視覺化比較)

### 1. 簡介
這是一個視覺化評估工具，專門用於**並排比較**立體匹配模型 (MSNet3D) 與單目深度估計模型 (Depth Anything 3) 的輸出差異。它解決了不同模型輸出格式不一（視差 vs 深度、近亮 vs 近黑）的問題，讓開發者能直觀地評估模型優劣。

**功能：**
* **三圖垂直對照**：將「RGB 原圖」、「MSNet3D 結果」、「Depth Anything 3 結果」由上而下排列，方便逐像素對照細節。
* **邏輯統一**：透過參數控制，將不同物理定義的數據（視差 Disparity / 深度 Depth）統一轉換為 **「近黑遠亮」** 的視覺邏輯。
* **自動增強對比**：內建百分位數截斷 (Percentile Clipping 1%~99%)，自動濾除極端噪點，避免整張圖黑成一片。

### 2. 使用方法

此腳本需要指定具體的 `.npy` 檔案路徑與對應的原始圖片。

1.  **編輯腳本**：打開腳本檔案，找到並修改以下變數：
    ```python
    # 設定檔案路徑
    image_num = '000080'
    img_path   = f'/path/to/image_2/{image_num}.png'      # RGB 原圖
    msnet_path = f'/path/to/msnet_result/{image_num}.npy' # MSNet3D 輸出的 .npy
    da3_path   = f'/path/to/da3_result/{image_num}.npy'   # Depth Anything 3 輸出的 .npy

    # 設定反轉邏輯 (目標：讓近處是黑色，遠處是亮色)
    # 如果執行後發現某個圖是 "近亮遠黑"，請將該項改成 True (或 False) 進行反轉
    INVERT_MSNET = False 
    INVERT_DA3   = False 
    ```

2.  **執行腳本**：
    ```bash
    python wen_compare_depth_models.py
    ```

### 3. 輸出解讀範例
程式將彈出一個視窗顯示三張圖，使用 **Magma 色票**（黑 -> 紫 -> 橘 -> 亮白）：

* **色彩檢查**：確認兩張深度圖的路面/近處物體皆為 **深色 (黑/紫)**，天空/遠景皆為 **亮色 (白/橘)**。若相反，請調整腳本中的 `INVERT` 參數。
* **MSNet3D (中間圖)**：
    * **優點**：具備真實物理尺度，幾何結構較準確。
    * **缺點**：注意觀察白牆或路面，可能會出現像雜訊般的顆粒 (Speckle noise)；物體邊緣可能有黑色遮擋區 (Occlusion)。
* **Depth Anything 3 (下方圖)**：
    * **優點**：畫面非常乾淨平滑，無雜訊，層次感強。
    * **缺點**：注意觀察細微物體（如欄杆、電線）是否消失；或物體邊緣是否過度銳利導致像「貼圖」一樣不自然。

### 4. 補充：詳細視覺比較指南

在使用本工具觀察結果時，建議專注於以下 5 個關鍵區域，這能幫助您快速判斷模型的特性與優劣：

#### 1. 弱紋理區域（白牆、路面、天空）
* **觀察重點**：找畫面中顏色單一、沒有花紋的地方（例如一面白牆或柏油路）。
* **MSNet3D (立體匹配)**：容易出現**破洞、雜訊或不規則的跳動**。因為它依賴左右影像的「特徵點」來配對，如果牆上什麼都沒有，它就無法計算，導致該區域深度值亂跳。
* **Depth Anything 3**：通常會表現得非常**平滑、乾淨**。因為它依靠語意理解（AI 認出這是牆），會自動把這塊區域塗平。
* **結論**：如果你需要畫面乾淨，**Depth Anything 3 勝**。

#### 2. 物體邊緣（遮擋區域）
* **觀察重點**：觀察物體與背景交界的地方（例如人的輪廓、車子的邊緣）。邊緣是否銳利？有沒有「殘影」？
* **MSNet3D**：在物體左側或右側容易出現**一圈無效值（黑邊）或模糊帶**。這是因為「左眼看得到、右眼被擋住」的物理死角（Occlusion）造成的。
* **Depth Anything 3**：邊緣通常切得**非常銳利**，很像修圖軟體摳圖的效果。但有時候會過度銳利，導致物體看起來像「紙板」貼在背景上。
* **結論**：**Depth Anything 3** 視覺上較討喜，但在邊緣處的深度可能不連續。

#### 3. 細微結構（欄杆、電線、樹枝）
* **觀察重點**：看路燈桿、細的樹枝是否斷斷續續，還是直接消失？
* **MSNet3D**：如果解析度夠高，它**有機會抓到真實的結構**，但容易斷掉。
* **Depth Anything 3**：經常會**直接忽略**細微物體，或者把它們變粗（為了讓畫面平滑），甚至把電線跟後面的天空黏在一起。
* **結論**：這部分兩者通常都不完美，但 **MSNet3D** 若調整得好，真實感會略高一點。

#### 4. 鏡面與透明物體（玻璃、水坑）
* **觀察重點**：看窗戶、鏡子或反光的地板。這是 AI 容易產生「幻覺」的地方。
* **MSNet3D**：測出的通常是**玻璃表面的距離**（如果有灰塵或反光），或者是雜訊。
* **Depth Anything 3**：很容易**看穿玻璃**，給出玻璃後方物體的深度；或者把鏡子裡的倒影當成真的空間（例如鏡子裡的人被標示在很遠的地方）。
* **結論**：如果你要做避障，**MSNet3D** 比較安全（它會認為那是障礙物）；Depth Anything 3 可能會讓你撞上玻璃。

#### 5. 遠處的漸層（天空與地平線）
* **觀察重點**：天空應該是無限遠，且深度變化應該很平順。
* **MSNet3D**：遠處視差極小，通常充滿雜訊，或者直接判定為無效值（0）。
* **Depth Anything 3**：能畫出非常漂亮的漸層，從近到遠很有層次感。
* **結論**：**Depth Anything 3** 在遠景表現完勝。

---
## 常見問題 (FAQ)

* **Q: `.npy` 和 `.pseudo` 檔案有什麼不同？**
    * **A:**
        * `.npy` 是 Float32 格式，未壓縮，檔案約 8.4MB。讀取方便 (`np.load`)。
        * `.pseudo` 是將 8 個體素狀態 (0/1) 壓縮進 1 個 Byte (uint8)，檔案約 262KB。讀取時需要位元解壓 (Unpack)。幾何資訊兩者完全相同。

* **Q: 執行時出現 `[WARN] Dependency gsplat is required...` 警告？**
    * **A:** 請忽略。這是 3D Gaussian Splatting 的依賴套件，我們只使用深度估計功能，不需要它。

* **Q: 輸出的深度圖大小與原圖不同？**
    * **A:** `preprocess_one_click.py` 已經內建自動 Resize 功能，推論後會強制將深度圖還原為原始影像的解析度，以確保點雲座標對齊，請放心使用。

* **Q: 程式執行後卡住沒有反應？**
    * **A:** 第一次執行時，程式需要下載並載入龐大的 `GIANT-LARGE` 模型到 GPU，這可能需要幾分鐘的時間，請耐心等待。