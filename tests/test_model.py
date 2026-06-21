import pytest
import torch
from src.models.segmentation_net import MultiResolutionFusionNet, UNetMultiFusion

class TestMultiResolutionFusionNet:
    def test_initialization(self):
        model = MultiResolutionFusionNet(num_classes=6, input_channels=[3, 8, 2], fusion_type='deep')
        assert model is not None
    
    def test_forward_pass(self):
        model = MultiResolutionFusionNet(num_classes=6, input_channels=[3, 8, 2])
        rgb = torch.randn(2, 3, 256, 256)
        ms = torch.randn(2, 8, 256, 256)
        sar = torch.randn(2, 2, 256, 256)
        output = model(rgb, ms, sar)
        assert output.shape == (2, 6, 256, 256)
    
    def test_different_fusion_types(self):
        for fusion_type in ['early', 'late', 'deep']:
            model = MultiResolutionFusionNet(fusion_type=fusion_type)
            rgb = torch.randn(1, 3, 256, 256)
            ms = torch.randn(1, 8, 256, 256)
            sar = torch.randn(1, 2, 256, 256)
            output = model(rgb, ms, sar)
            assert output.shape == (1, 6, 256, 256)

class TestUNetMultiFusion:
    def test_initialization(self):
        model = UNetMultiFusion(num_classes=6)
        assert model is not None
    
    def test_forward_pass(self):
        model = UNetMultiFusion()
        rgb = torch.randn(2, 3, 256, 256)
        ms = torch.randn(2, 8, 256, 256)
        sar = torch.randn(2, 2, 256, 256)
        output = model(rgb, ms, sar)
        assert output.shape == (2, 6, 256, 256)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
