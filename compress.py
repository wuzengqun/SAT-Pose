import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = PoseCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = PoseFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'model/best.pt',
        'data':'',
        'imgsz': 640,
        'epochs': 1200,
        'batch': 32,
        'workers': 4,
        'cache': False,
        'optimizer': 'SGD',
        'device': 'cpu',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov11-pose-prune_test',
        
        # prune
        'prune_method':'LAMP',
        'global_pruning': True,
        'speed_up': 1.5,
        # 'reg': 0.0005,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)