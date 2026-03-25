from utils.video_augmentation import *
# Thêm comment cho các class bị thiếu để tránh lỗi ImportError
from dataset.Uniformer_dataset import UFOneView_Dataset #, UFThreeView_Dataset
from dataset.MaskUniformer_dataset import MaskUFOneView_Dataset #, MaskUFThreeView_Dataset

def build_dataset(dataset_cfg, split, model=None, **kwargs):
    dataset = None
    
    # 1. Xử lý các model OneView
    if dataset_cfg['model_name'] in ['UFOneView', 'mvit_v2', 'swin']:
        dataset = UFOneView_Dataset(dataset_cfg['base_url'], split, dataset_cfg, **kwargs)
    
    if dataset_cfg['model_name'] == 'MaskUFOneView':
        dataset = MaskUFOneView_Dataset(dataset_cfg['base_url'], split, dataset_cfg, **kwargs)

    # 2. Xử lý các model yêu cầu ThreeView (UsimKD)
    # Vì file của bạn thiếu UFThreeView_Dataset, chúng ta tạm thời dùng UFOneView_Dataset để chạy test
    if dataset_cfg['model_name'] in ['UFThreeView', 'UsimKD']:
        print(f"--> Warning: {dataset_cfg['model_name']} requires ThreeView, but falling back to OneView.")
        dataset = UFOneView_Dataset(dataset_cfg['base_url'], split, dataset_cfg, **kwargs)

    assert dataset is not None, f"Dataset model_name '{dataset_cfg['model_name']}' is not supported or class is missing."
    return dataset