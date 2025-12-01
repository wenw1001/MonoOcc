# 快速開始

單張影像的完整推論流程：
如果要將一張全新的、沒看過的影像丟入模型得出結果，你需要建立一個完整的自動化管線 (Pipeline)，依序執行以下步驟：
1. Stage 0: 前處理 (深度估計)
- 輸入：單張 RGB 影像 (image.png)
- 執行：執行深度估計模型 (如 MobileStereoNet3D 或 DA3)。
- 產出：深度圖 -> 偽點雲 -> 偽體素 (Pseudo Voxel .npy)。

2. Stage 1: 提案生成 (QPN 推論)
- 輸入：上一步產生的 偽體素 (.npy)。
- 執行：QPN 模型 (qpn.pth)。
- 產出：稀疏提案 (.query 或 .npy)。

3. Stage 2: 語意補全 (MonoOcc 推論)
- 輸入：原始 RGB 影像 (image.png) + 上一步產生的 稀疏提案 (.query)。
- 執行：MonoOcc 模型 (MonoOcc-S.pth)。
- 產出：最終 3D 語意地圖 (.pkl 或 ssc_logit)。

以下是三階段指令，須先準備好 `影像資料夾` 以及 `calib.txt`

-----
## Stage 0 前處理
####　啟動環境
```bash
cd Desktop/wenwen/my_projects/MonoOcc/

conda activate da3
```

#### 執行編號00的資料夾
```bash
python /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click.py \
    --image-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/image_2 \
    --calib-path /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/calib.txt \
    --save-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00 \
    --save-depth \
    --save-depth-vis \
    --save-lidar \
    --compress
```

---
## Stage 1 QPN
####　啟動環境
```bash
conda deactivate
conda activate MonoOcc_clean
```

#### 執行編號00的資料夾
```bash
# cd ./MonoOcc
python ./tools/wen_test_stage1.py \
    ./projects/configs/MonoOcc/qpn.py \
    /home/rvl/Desktop/wenwen/my_projects/MonoOcc/ckpts/qpn_mIoU_6140_epoch_12.pth \
    ./wen_data/wen_kitti/00
```

---
## Stage 2 MonoOcc
#### 執行編號00的資料夾
```bash
# cd ./MonoOcc
python ./tools/wen_test_stage2.py \
    ./projects/configs/MonoOcc/MonoOcc-S.py \
    /home/rvl/Desktop/wenwen/my_projects/MonoOcc/ckpts/MonoOcc-S.pth \
    ./wen_data/wen_kitti/00
```





















---
## 其他

### Stage 0 原始官方版本
####　安裝環境：
1. 建立 `mobilestereonet_py310`
```bash
# 切到 MonoOcc 底下 preprocess 目錄
cd /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess
# 升級 PyTorch + CUDA runtime 到支援 RTX 4090 的版本
conda create -n mobilestereonet_py310 python=3.10
conda activate mobilestereonet_py310
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade typing_extensions
pip install opencv-python
```

2. 建立 `mono_preprocess`
```bash
conda deactivate

# 建 env
conda create -n mono_preprocess python=3.7 -y
conda activate mono_preprocess

# 安裝簡單套件（若 conda 裡面沒有可改用 pip）
conda install -y numpy tqdm pyyaml imageio || pip install numpy tqdm pyyaml imageio

# 回到 MonoOcc repo 的 mapping (路徑視 repo 而定)
cd /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mapping

# 安裝系統相依（Ubuntu）
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev

rm -rf build          # 刪掉舊 build
rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

conda install -c conda-forge pybind11

# 建 build 並編譯
mkdir -p build && cd build
cmake ..
make -j$(nproc) # 同指令：make all
# 將 build 路徑加入 Python 路徑(有剛編譯好的 Python extension .so 檔案)
export PYTHONPATH=/home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mapping/build:$PYTHONPATH
```

#### 執行：
1. depth2lidar
```bash
cd /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess
conda activate mobilestereonet_py310
./depth2lidar.sh # depth maps 產生後，轉成 pseudo point cloud，生成 lidar 資料夾(會在/home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mobilestereonet/lidar目錄下)
```
或直接執行以下指令
```bash
# depth2lidar.sh 內容如下
set -e
exeFunc(){
    # num_seq=$1
    python utils/depth2lidar.py --calib_dir  /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00 \
    --depth_dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/depth_da3/ \
    --save_dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar/

    cp /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/calib.txt /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar/
    cp /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/poses.txt /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar/
}

exeFunc
```

2. lidar2voxel
下面步驟為 Point cloud → Voxel（mapping / lidar2voxel）
```bash
conda deactivate
conda activate mono_preprocess

cd /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess
bash ./lidar2voxel.sh
```
或執行以下指令
```bash
# lidar2voxel.sh內容如下，資料夾選用00
set -e
exeFunc(){
    num_seq=$1
    sequence_length=$2
    python utils/lidar2voxel.py \
    --dataset /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/lidar/ \
    --output /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/00/pseudo_da3_sh \
    --num_seq $num_seq \
    --sequence_length $sequence_length
}
exeFunc 00 1
```
lidar2voxel.sh 會把 pseudo-lidar（depth2lidar 的輸出）合併 / sweep，並輸出 voxel，最終放到：`/home/rvl/Desktop/wenwen/kitti/dataset/sequences_msnet3d_sweep10/00/voxels/*.pseudo`