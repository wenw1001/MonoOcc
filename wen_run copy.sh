#!/bin/bash

# --- 設定變數 (方便未來修改) ---
PROJECT_ROOT="/home/rvl/Desktop/wenwen/my_projects/MonoOcc"
DATA_FILE="06" # 修改為你的資料集名稱
DATA_ROOT="${PROJECT_ROOT}/kitti/dataset/sequences/${DATA_FILE}"
PREPROCESS_SCRIPT="${PROJECT_ROOT}/wen_data/preprocess/preprocess_one_click.py" # 預處理腳本路徑
VISUAL_SCRIPT="${PROJECT_ROOT}/wen_data/visual_pkl_snapshot.py" # 可視化腳本路徑
VIDEO_SCRIPT="${PROJECT_ROOT}/wen_data/wen_result_video.py" # 生成影片腳本路徑
START_INDEX=0 # 起始拍攝索引
END_INDEX=4800 # 結束拍攝索引 (根據你的資料集調整)

# 設定起始版本號
START_VERSION=0 

# --- 檢查是否開啟清潔模式 ---
# 檢查腳本的第一個參數是否為 "--clean"
CLEAN_MODE=false
if [[ "$1" == "--clean" ]]; then
    CLEAN_MODE=true
fi

# 讓腳本遇到錯誤就停止
set -e

echo "========================================================"
echo "   開始執行 MonoOcc 全自動流程"
if [ "$CLEAN_MODE" = true ]; then
    echo "   [模式] 清潔模式：將刪除舊的 queries 資料夾"
else
    echo "   [模式] 一般模式：保留並覆蓋舊 queries"
fi
echo "資料夾: ${DATA_ROOT}"
echo "========================================================"

# # --- 初始化 Conda ---
eval "$(conda shell.bash hook)"

# ========================================================
# Step 0: 清除舊資料 (僅在 --clean 模式下執行)
# 注意：必須在 Step 1 之前執行，否則會把 Step 1 剛生成的檔案刪掉
# ========================================================
if [ "$CLEAN_MODE" = true ]; then
    echo "[Step 0/5] 清除舊資料..."
    echo "  - 刪除 queries_2, queries_4, queries_8"
    rm -rf "${DATA_ROOT}/queries_2"
    rm -rf "${DATA_ROOT}/queries_4"
    rm -rf "${DATA_ROOT}/queries_8"
    
    echo "  - 刪除 pseudo_packed"
    rm -rf "${DATA_ROOT}/pseudo_packed"
    
    echo "清除完成，準備開始生成新資料。"
fi

# ========================================================
# Step 1: 執行 Preprocess (環境: da3)
# ========================================================
echo "[Step 1/5] 切換至 da3 環境，執行預處理..."
conda activate da3

python "$PREPROCESS_SCRIPT" \
    --image-dir "${DATA_ROOT}/image_2" \
    --calib-path "${DATA_ROOT}/calib.txt" \
    --save-dir "${DATA_ROOT}" \
    --compress
#   --save-depth \
#   --save-depth-vis \
#   --save-lidar \
#     # --manual-pitch -20.0 \
#     # --manual-yaw -9.0 \
#     # --manual-height 3 \
#     # --focal-scale 0.8 \
#     # --z-stretch 2.0 \


# python /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click_v3.py \
#     --image-dir "${DATA_ROOT}/image_2" \
#     --calib-path "${DATA_ROOT}/calib.txt" \
#     --save-dir "${DATA_ROOT}" \
#     --save-depth \
#     --save-depth-vis \
#     --save-lidar \
#     --compress \
#     --manual-height 1.73 \
#     --focal-scale 0.85 \
#     --max-height 4

# python "/home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click_v2.py" \
#     --image-dir "${DATA_ROOT}/image_2" \
#     --calib-path "${DATA_ROOT}/calib.txt" \
#     --save-dir "${DATA_ROOT}" \
#     --save-lidar \
#     --compress \
#     --cut-top 0.0 \
#     --cut-bottom 0.0 \
#     --manual-height 1.5 \
#     --focal-scale 1.35 \
#     --pitch -2.0

# python /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/preprocess/preprocess_one_click_v2.py \
#     --image-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00/image_2 \
#     --calib-path /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00/calib.txt \
#     --save-dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/wen_data/wen_kitti/test_00 \
#     --save-lidar \
#     --compress





# echo "預處理完成。"

# ========================================================
# Step 2: 執行兩階段預測 (環境: monoOcc_clean)
# ========================================================
echo "[Step 2/5] 切換至 monoOcc_clean 環境，執行 Stage 1 & 2..."
conda activate MonoOcc_clean
cd "$PROJECT_ROOT"

# Stage 1
echo "Running Stage 1..."
python ./tools/wen_test_stage1.py \
    ./projects/configs/MonoOcc/qpn.py \
    "${PROJECT_ROOT}/ckpts/qpn_mIoU_6140_epoch_12.pth" \
    "${DATA_ROOT}"

# Stage 2
echo "Running Stage 2..."
python ./tools/wen_test_stage2.py \
    ./projects/configs/MonoOcc/MonoOcc-S.py \
    "${PROJECT_ROOT}/ckpts/MonoOcc-S.pth" \
    "${DATA_ROOT}" 

echo "預測完成，.pkl 已生成。"

# # ========================================================
# # Step 3: 視覺化 .pkl (環境: monoOcc_clean)
# # ========================================================
# echo "[Step 3/5] 執行視覺化 (PKL -> Images)..."
# # 假設視覺化腳本也在 monoOcc_clean 環境下執行
# python "$VISUAL_SCRIPT" --dir "${DATA_ROOT}/predictions" --seq "${DATA_FILE}" --start "$START_INDEX" --end "$END_INDEX"

# # ========================================================
# # Step 4: 製作影片 (環境: monoOcc_clean)
# # ========================================================
# echo "[Step 4/5] 製作影片 (Images -> Video)..."
# python "$VIDEO_SCRIPT" --rgb "${DATA_ROOT}/image_2" --result "${DATA_ROOT}/predictions/3D_Map_Screenshots"

# ========================================================
# Step 5: 自動更改資料夾名稱 (predictions -> predictions_vXX)
# ========================================================
echo "[Step 5/5] 版本號管理..."

cd "$DATA_ROOT"

# 檢查 predictions 資料夾是否存在
if [ ! -d "predictions" ]; then
    echo "警告: 找不到 'predictions' 資料夾，跳過更名步驟。"
else
    # 邏輯：找出目前存在的 predictions_v??，取最大值
    max_ver=$START_VERSION
    
    # 遍歷所有符合 predictions_v* 的資料夾
    for d in predictions_v*; do
        if [ -d "$d" ]; then
            # 取出 v 後面的數字 (例如 predictions_v10 -> 10)
            ver=${d#predictions_v}
            # 如果這個數字比目前記錄的大，就更新 max_ver
            if [[ "$ver" =~ ^[0-9]+$ ]] && [ "$ver" -gt "$max_ver" ]; then
                max_ver=$ver
            fi
        fi
    done

    # 下一個版本號
    next_ver=$((max_ver + 1))
    new_name="predictions_v${next_ver}"

    echo "檢測到最新版本為 v${max_ver}，本次結果將命名為: ${new_name}"
    
    # 執行更名 (predictions)
    mv predictions "$new_name"
    echo "已將 predictions 更名為 ${new_name}"

    # [新增] 同步更名 lidar_da3
    if [ -d "lidar_da3" ]; then
        mv lidar_da3 "lidar_da3_v${next_ver}"
        echo "已將 lidar_da3 更名為 lidar_da3_v${next_ver}"
    fi

    # [新增] 同步更名 pseudo_packed
    if [ -d "pseudo_packed" ]; then
        mv pseudo_packed "pseudo_packed_v${next_ver}"
        echo "已將 pseudo_packed 更名為 pseudo_packed_v${next_ver}"
    fi
fi

echo "========================================================"
echo "   所有流程執行完畢！"
echo "========================================================"
