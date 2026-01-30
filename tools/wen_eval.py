"""
使用方式：
python ./tools/wen_eval.py ./projects/configs/MonoOcc/MonoOcc-S.py result/MonoOcc-S/某次訓練的資料夾
"""
import argparse
import mmcv
import torch
import numpy as np
import sys
import os
import re
import gc
import time  # 用來生成時間戳記

# [Fix] 將腳本的上一層目錄 (專案根目錄) 加入 Python 搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from projects.mmdet3d_plugin.datasets import semantic_kitti_dataset_stage2

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader
from mmcv.parallel import collate, scatter

# 解決 OOM 關鍵
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Memory Efficient Evaluation for MonoOcc')
    parser.add_argument('config', help='config file path (e.g., MonoOcc-S.py)')
    parser.add_argument('checkpoint_path', help='directory containing checkpoints OR specific .pth file')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    args = parser.parse_args()
    return args

# [New] 定義一個雙重輸出的 helper function
def log_msg(msg, log_file=None):
    """同時 print 到終端機並寫入檔案"""
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush() # 確保即時寫入，防止斷電資料遺失

def single_gpu_test_and_evaluate(model, data_loader, dataset, epoch_name, log_file=None):
    model.eval()
    
    # 進度條只顯示在 Terminal，不寫入 Log
    print(f"\n[Evaluating {epoch_name}] Progress:")
    prog_bar = mmcv.ProgressBar(len(data_loader))
    
    if hasattr(dataset, 'metrics'):
        dataset.metrics.reset()
    else:
        log_msg("Error: Dataset does not have 'metrics' attribute.", log_file)
        return

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # 1. 搬運數據到 GPU
            target_device_id = next(model.parameters()).device.index
            data = scatter(data, [target_device_id])[0]
            
            # 2. 推論
            result = model(return_loss=False, rescale=True, **data)
            
            # [Fix] 格式與維度處理
            if isinstance(result, list):
                result = result[0]
            
            if 'y_pred' in result:
                pred_occ = result['y_pred']
            elif 'output_voxels' in result:
                pred_occ = result['output_voxels']
            else:
                continue

            if isinstance(pred_occ, list):
                pred_occ = pred_occ[0]
            
            if isinstance(pred_occ, torch.Tensor):
                if pred_occ.ndim == 5 and pred_occ.shape[0] == 1:
                    pred_occ = pred_occ.squeeze(0)
                if pred_occ.ndim == 4:
                    pred_occ = pred_occ.argmax(dim=0)
                y_pred = pred_occ.cpu().numpy().astype(np.uint8)
            elif isinstance(pred_occ, np.ndarray):
                y_pred = pred_occ.astype(np.uint8)
            else:
                continue

            # 3. 獲取 Ground Truth
            img_metas = data['img_metas'][0][0]
            
            if 'sequence' in img_metas:
                sequence = img_metas['sequence']
            elif 'sequence_id' in img_metas:
                sequence = img_metas['sequence_id']
                if isinstance(sequence, int):
                    sequence = str(sequence).zfill(2)
                elif isinstance(sequence, str) and len(sequence) == 1 and sequence.isdigit():
                     sequence = sequence.zfill(2)
            else:
                continue 

            frame_id = img_metas['frame_id']
            if isinstance(frame_id, int):
                 frame_id = str(frame_id).zfill(6)

            try:
                target, target_2, target_4, target_8 = dataset.get_gt_info(sequence, frame_id)
                y_true = target.astype(np.uint8)
            except Exception as e:
                log_msg(f"\nError reading GT: {e}", log_file)
                continue

            if y_pred.shape != y_true.shape:
                if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[0] == 1:
                    y_pred = y_pred.squeeze(0)
                if y_pred.shape != y_true.shape:
                    log_msg(f"\nShape Mismatch! Pred: {y_pred.shape}, GT: {y_true.shape}", log_file)
                    continue

            # 4. 累積計算
            dataset.metrics.add_batch(y_pred, y_true)

            del result, pred_occ, y_pred, y_true, target
        
        prog_bar.update()

    # [New] 使用 log_msg 替代 print，將結果寫入檔案
    log_msg(f"\n\n=== Results for {epoch_name} ===", log_file)
    
    metric_prefix = 'ssc_SemanticKITTI'
    stats = dataset.metrics.get_stats()
    
    log_msg("\n====== Per Class IoU ======", log_file)
    for i, class_name in enumerate(dataset.class_names):
        iou = stats["iou_ssc"][i]
        log_msg(f"{class_name:<20}: {iou:.4f}", log_file)
    
    log_msg("===========================", log_file)
    log_msg(f"mIoU (Mean IoU)   : {stats['iou_ssc_mean']:.4f}", log_file)
    log_msg(f"IoU (Occupancy)   : {stats['iou']:.4f}", log_file)
    log_msg(f"Road IoU          : {stats['iou_ssc'][9]:.4f}", log_file)
    log_msg(f"Precision         : {stats['precision']:.4f}", log_file)
    log_msg(f"Recall            : {stats['recall']:.4f}", log_file)
    log_msg("===========================", log_file)

    dataset.metrics.reset()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Importing plugin from: {_module_path}")
                plg_lib = importlib.import_module(_module_path)

    dataset = build_dataset(cfg.data.val)
    
    data_loader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate, 
        pin_memory=False
    )

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = model.cuda(args.gpu_id)

    # 4. 處理 Checkpoint 列表與 Log 路徑
    pth_files = []
    log_dir = ""
    
    if os.path.isdir(args.checkpoint_path):
        print(f"Scanning checkpoints in {args.checkpoint_path}...")
        log_dir = args.checkpoint_path # Log 存在跟 checkpoint 同一層
        files = os.listdir(args.checkpoint_path)
        pth_files = [f for f in files if f.endswith('.pth') and 'epoch_' in f]
        pth_files.sort(key=lambda x: int(re.findall(r'epoch_(\d+).pth', x)[0]))
        pth_files = [os.path.join(args.checkpoint_path, f) for f in pth_files]
    
    elif os.path.isfile(args.checkpoint_path):
        log_dir = os.path.dirname(args.checkpoint_path) # Log 存在檔案的資料夾
        pth_files = [args.checkpoint_path]
    
    else:
        print("Error: Invalid checkpoint path.")
        return

    # [New] 建立 Log 檔案
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file_path = os.path.join(log_dir, f"eval_results_{timestamp}.txt")
    
    print(f"Found {len(pth_files)} checkpoints.")
    print(f"Results will be saved to: {log_file_path}")

    # 開啟檔案準備寫入
    with open(log_file_path, 'w') as log_file:
        log_msg(f"Evaluation started at {timestamp}", log_file)
        log_msg(f"Config: {args.config}", log_file)
        log_msg("-" * 50, log_file)

        # 5. 批次迴圈驗證
        for pth in pth_files:
            epoch_name = os.path.basename(pth)
            log_msg(f"\nProcessing: {epoch_name}", log_file)
            
            try:
                load_checkpoint(model, pth, map_location='cpu')
                
                # 傳入 log_file 物件
                single_gpu_test_and_evaluate(model, data_loader, dataset, epoch_name, log_file)
                
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                log_msg(f"Failed to evaluate {epoch_name}: {e}", log_file)
                continue
    
    print(f"\nAll evaluations finished. Log saved to {log_file_path}")

if __name__ == '__main__':
    main()