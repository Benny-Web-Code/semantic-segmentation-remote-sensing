import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RemoteSensingDataset(Dataset):
    def __init__(self, rgb_dir, ms_dir, sar_dir, mask_dir, img_size=(512, 512), augment=True):
        self.rgb_dir = Path(rgb_dir)
        self.ms_dir = Path(ms_dir)
        self.sar_dir = Path(sar_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment
        
        self.rgb_files = sorted(list(self.rgb_dir.glob('*.tif')) + list(self.rgb_dir.glob('*.png')))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        base_name = rgb_path.stem
        
        rgb = self._load_image(self.rgb_dir / f"{base_name}.tif", 3)
        ms = self._load_image(self.ms_dir / f"{base_name}.tif", 8)
        sar = self._load_image(self.sar_dir / f"{base_name}.tif", 2)
        mask = self._load_image(self.mask_dir / f"{base_name}.tif", 1)
        
        rgb = self._normalize(rgb, 'rgb')
        ms = self._normalize(ms, 'multispectral')
        sar = self._normalize(sar, 'sar')
        
        rgb = torch.from_numpy(rgb).float()
        ms = torch.from_numpy(ms).float()
        sar = torch.from_numpy(sar).float()
        mask = torch.from_numpy(mask).long()
        
        return {
            'rgb': rgb,
            'multispectral': ms,
            'sar': sar,
            'mask': mask.squeeze(),
            'filename': base_name
        }

    def _load_image(self, path, num_channels):
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path)
        data = np.array(img)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        else:
            data = np.transpose(data, (2, 0, 1))
        
        return data.astype(np.float32)

    def _normalize(self, image, image_type):
        if image_type == 'rgb':
            image = np.clip(image / 255.0, 0, 1)
        else:
            mean = image.mean()
            std = image.std() + 1e-8
            image = (image - mean) / std
        return image
