"""
此為將MonoOcc給的程式自行簡化後的test.py，設計用於單卡測試 (1 GPU)、強制使用分散式環境、沒有驗證 (eval) 階段
需要修改的部份為Config 路徑、Checkpoint 路徑、欲推論影像檔路徑

資料根目錄 結構應該如下：
    data_root/
    ├── calib.txt
    ├── poses.txt
    ├── image_2/
    │   ├── 000000.png ...
    ├── voxels/
    │   ├── 000000.pseudo ...
    └── queries_2/ (或其他你存放 proposal 的路徑)
        ├── 000000.proposal ...
        

使用方法，打開終端機輸入以下指令：
cd ./MonoOcc
python ./tools/wen_test_stage2.py \
    ./projects/configs/MonoOcc/MonoOcc-S.py \
    /home/rvl/Desktop/wenwen/my_projects/MonoOcc/ckpts/MonoOcc-S.pth \
    ./wen_data/wen_kitti/00

"""

import argparse
import os
import sys
sys.path.append('.')
import torch
import warnings
from functools import partial
import numpy as np
import random
import time
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDistributedDataParallel, collate
from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model, get_dist_info
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from torch.utils.data import DataLoader

# 自定義模組 import
from projects.mmdet3d_plugin.datasets.wen_self_kitti_dataset2 import SelfKittiDatasetStage2
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler
from projects.mmdet3d_plugin.MonoOcc.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    # 你的 Config 路徑
    default_config = './projects/configs/MonoOcc/MonoOcc-S.py'
    # 你的 Checkpoint 路徑
    default_ckpt = '/home/rvl/Desktop/wenwen/my_projects/MonoOcc/ckpts/MonoOcc-S.pth'
    default_data_root = './wen_data/wen_kitti/07'

    # nargs='?' 代表如果指令沒給這個參數，就使用 default 值
    parser.add_argument('config', nargs='?', default=default_config, help='test config file path')
    parser.add_argument('checkpoint', nargs='?', default=default_ckpt, help='checkpoint file')
    parser.add_argument('data', nargs='?', default=default_data_root, help='checkpoint file')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config')
    # ----------------------------------------

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # --- 新增這段：自動設定單機單卡環境變數 ---
    if 'RANK' not in os.environ:
        print("偵測到未使用 distributed launch 啟動，自動設定為單機單卡模式...")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    
    print('-----------------')
    print('1. 參數設定:')
    args = parse_args()
    print(args)

    print('-----------------')
    print('2. 載入配置檔案...')
    # 1. 載入配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 處理 plugin import
    if hasattr(cfg, 'plugin') and cfg.plugin:
        print("有 plugin，開始 import...")
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    print("配置檔案載入完成。")
    print("cfg 設定內容:")
    time.sleep(1)  
    print(cfg.pretty_text)
    time.sleep(1)  

    # 2. 環境初始化
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 3. 建立 Dataset (直接使用指定的 Class)
    print('-----------------')
    print("3. 從 SimpleKittiDataset 建立資料集物件...")
    dataset = SelfKittiDatasetStage2(
        data_root=args.data,
        semantickitti_yaml="/home/rvl/Desktop/wenwen/kitti/dataset/semantic-kitti.yaml",
        labels_filename="labels",
        query_filename="queries_2",
        query_tag="query"
    )
    print("dataset 物件建立完成，包含 %d 筆資料。" % len(dataset))
    time.sleep(1)  # for better print output

    # 4. 建立 DataLoader (手動建置 Sampler 與 Loader)
    print('-----------------')
    print("4. 建立資料載入器（dataloader）...")
    shuffle = False
    num_workers = cfg.data.workers_per_gpu
    nonshuffler_sampler = cfg.data.nonshuffler_sampler
    
    sampler = build_sampler(
        nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
        dict(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=args.seed
        )
    )

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=args.seed) if args.seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn
    )
    print("資料載入器 (dataloader) 建立完成。")

    # 5. 建立與載入模型
    cfg.model.train_cfg = None
    print('-----------------')
    print("5-1. 建立模型...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print("模型建立完成。")
    # print("模型結構:")
    # print(model)
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    print('-----------------')
    print("5-2. 載入檢查點檔案...")
    print("checkpoint路徑: ", args.checkpoint)
    time.sleep(1)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    # 6. 執行測試
    print('-----------------')
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False
    )
    start_time = time.time()
    print(f"開始多重 GPU 推論..., 開始時間: {time.ctime()}")
    outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    print(f"\n多重 GPU 推論完成。共有 {len(outputs)} 筆輸出結果。費時: {time.time() - start_time} 秒。結束時間: {time.ctime()}")


if __name__ == '__main__':
    main()