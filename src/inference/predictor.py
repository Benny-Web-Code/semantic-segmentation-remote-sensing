import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model, checkpoint_path=None, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        logger.info("Predictor initialized")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def predict(self, rgb, multispectral, sar, return_probabilities=False):
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        ms_tensor = torch.from_numpy(multispectral).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        sar_tensor = torch.from_numpy(sar).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            output = self.model(rgb_tensor, ms_tensor, sar_tensor)
        
        if return_probabilities:
            return torch.softmax(output, dim=1).squeeze().cpu().numpy()
        else:
            return output.argmax(dim=1).squeeze().cpu().numpy()
