import importlib.metadata
import json
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from railflaw_seg.runtime import PROJECT_ROOT, configure_runtime


if __name__ == '__main__':
    configure_runtime()
    import torch
    report = {'project': str(PROJECT_ROOT), 'cuda_available': torch.cuda.is_available(),
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'torch_cuda_build': torch.version.cuda,
              'versions': {p: importlib.metadata.version(p) for p in ['ultralytics', 'torch', 'torchvision', 'opencv-python', 'numpy', 'Pillow']},
              'prepared_data': Path('D:/dataset/rail/yolo/data.yaml').exists()}
    destination = PROJECT_ROOT / 'reports/environment.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), 'utf-8')
    print(json.dumps(report, indent=2))
