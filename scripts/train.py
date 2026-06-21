import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import RemoteSensingDataset
from src.models.segmentation_net import MultiResolutionFusionNet
from src.training.trainer import Trainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    if args.epochs:
        config.config['training']['epochs'] = args.epochs
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_dataset = RemoteSensingDataset(
        rgb_dir=str(Path(config.data['train_dir']) / 'images' / 'rgb'),
        ms_dir=str(Path(config.data['train_dir']) / 'images' / 'multispectral'),
        sar_dir=str(Path(config.data['train_dir']) / 'images' / 'sar'),
        mask_dir=str(Path(config.data['train_dir']) / 'masks'),
        img_size=tuple(config.data['img_size']),
        augment=True
    )
    
    val_dataset = RemoteSensingDataset(
        rgb_dir=str(Path(config.data['val_dir']) / 'images' / 'rgb'),
        ms_dir=str(Path(config.data['val_dir']) / 'images' / 'multispectral'),
        sar_dir=str(Path(config.data['val_dir']) / 'images' / 'sar'),
        mask_dir=str(Path(config.data['val_dir']) / 'masks'),
        img_size=tuple(config.data['img_size']),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.training['batch_size'], shuffle=True, num_workers=config.data['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config.training['batch_size'], num_workers=config.data['num_workers'])
    
    model = MultiResolutionFusionNet(
        num_classes=config.model['num_classes'],
        input_channels=config.model['input_channels'],
        fusion_type=config.model['fusion_type'],
        backbone=config.model['backbone'],
        use_attention=config.model['use_attention']
    )
    
    trainer = Trainer(model, train_loader, val_loader, config.config, device)
    trainer.train()

if __name__ == '__main__':
    main()
