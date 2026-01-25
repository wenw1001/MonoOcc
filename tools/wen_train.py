# -------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# -------------------------------------------------------------------
#  Modified by Yiming Li
# -------------------------------------------------------------------

from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    print('-----------------')
    print('1. 參數設定:')
    args = parse_args()
    print(args)

    # 1. 讀取並合併配置檔案 (MonoOcc-S.py)
    print('-----------------')
    print('2. 載入配置檔案...')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 2. 處理自定義導入 (Custom Imports)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # ============================================================
    # [關鍵點 1] Plugin 動態加載機制
    # MonoOcc 不是 MMDetection3D 原生支援的模型，它是透過 plugin 形式注入的。
    # 這段程式碼會去尋找 'projects/mmdet3d_plugin/' 目錄，
    # 並載入那裡的代碼（例如 MonoOcc-S.py 裡定義的 MonoOccHead）。
    # ============================================================
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            print("有 plugin，開始 import...")
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
                
            # [重要] 從 Plugin 中導入客製化的訓練函數
            # 這意味著它不用 mmdet3d 預設的 train_model，而是用自己寫的 custom_train_model
            # 這通常是因為 Occupancy 任務需要特殊的 loss 計算或視覺化流程。
            from projects.mmdet3d_plugin.MonoOcc.apis.train import custom_train_model

    # 設定 CUDNN (加速訓練)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    print("配置檔案載入完成。")

    # 3. 設定工作目錄 (Work Directory)
    # 訓練好的模型 (.pth) 和 Log 會存在這裡。預設是 ./result/MonoOcc/MonoOcc-S
    # work_dir is determined in this priority: CLI > segment in file > filename
    print('-----------------')
    print('3. 設定工作目錄...')
    # [Modify] 自動建立帶有時間戳記的獨立資料夾
    # 格式範例: result/MonoOcc-S/20260123_214530/
    
    # 1. 取得當前時間字串 (例如: 20260123_214530)
    timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # 2. 決定基礎路徑 (Base Path)
    if args.work_dir is not None:
        base_work_dir = args.work_dir
    elif cfg.get('work_dir', None) is not None:
        base_work_dir = cfg.work_dir
    else:
        # 預設路徑: ./result/MonoOcc/config檔名
        base_work_dir = osp.join('./result/MonoOcc',
                                 osp.splitext(osp.basename(args.config))[0])
    
    # 3. 組合最終路徑: 基礎路徑 + 時間戳記
    cfg.work_dir = osp.join(base_work_dir, timestamp_str)
        
    # 斷點續訓 (Resume) 設定
    # if args.resume_from is not None:
    print('-----------------')
    print('4. 設定斷點續訓...')
    if args.resume_from is not None and osp.isfile(args.resume_from):
        print(f"從 {args.resume_from} 繼續訓練")
        cfg.resume_from = args.resume_from

    # GPU ID 設定
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # [Bug Fix] 針對特定 PyTorch 版本的 AdamW 優化器修復
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw

    # 自動調整學習率 (Learning Rate Scaling)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # 4. 初始化分散式訓練環境 (Distributed Training)
    # 這是多 GPU 訓練的核心，透過 NCCL 進行通訊。
    # init distributed env first, since logger depends on the dist info.
    print('-----------------')
    print('5. 初始化分散式訓練環境...')
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 建立目錄並保存當前 Config (備份用，這很重要，以後才知道這個模型是用什麼參數練的)
    print('-----------------')
    print('6. 建立目錄並保存當前 Config...')
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    print(f"當前配置已保存。 儲存在 {osp.join(cfg.work_dir, osp.basename(args.config))}")

    # 5. 初始化 Logger (記錄器)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        print("model type 為 EncoderDecoder3D，設定 logger_name 為 mmseg")
        logger_name = 'mmseg'
    else:
        print("model type 為 非 EncoderDecoder3D，設定 logger_name 為 mmdet")
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # 記錄環境資訊 (PyTorch版本, CUDA版本等) -> 除錯關鍵
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}') # 在 log 中印出完整參數

    # 設定隨機種子 (Seed) 以確保結果可重現 (Reproducibility)
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # ============================================================
    # [關鍵點 2] 建立模型 (Build Model)
    # 這裡會根據 MonoOcc-S.py 中的 model = dict(type='MonoOcc', ...) 
    # 來實例化模型物件。如果這裡報錯，通常是 Plugin 沒載入成功，
    # 或者 Config 裡的 type 名稱寫錯。
    # ============================================================
    print('-----------------')
    print('7. 建立模型...')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    print(f"模型建立完成。")
    # ============================================================
    # [關鍵點 3] 建立數據集 (Build Dataset)
    # 這裡會呼叫 SemanticKittiDatasetStage2。
    # 這是你未來要換成 ITRI 或 Mio 數據時，必須修改或替換的核心部分。
    # ============================================================
    datasets = [build_dataset(cfg.data.train)]

    # 處理驗證集 (Validation Set)
    print('-----------------')
    print('8. 處理驗證集...')
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    # 儲存 Checkpoint 的 Meta 資訊
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text)
            # CLASSES=datasets[0].CLASSES,
            # PALETTE=datasets[0].PALETTE  # for segmentors
            # if hasattr(datasets[0], 'PALETTE') else None
    # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES

    # ============================================================
    # [關鍵點 4] 開始訓練
    # 呼叫 plugin 裡的 custom_train_model。
    # 這函數會處理 Epoch 迴圈、Loss Backward、Optimizer Step 等細節。
    # ============================================================
    print('-----------------')
    print('9. 開始訓練...')
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
    print('訓練完成。')

if __name__ == '__main__':
    main()
