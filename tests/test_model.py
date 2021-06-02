
import torch
import torch.nn.functional as F
import torch_ihc_segmentation as TIS

def test_model_returns_correct_num_classes():
    sample_image = torch.rand([4,3,112,112])

    for i in range(5):
        mod = TIS.models.LitSegmentationModel(pretrained=False, num_classes=i+1)
        assert mod(sample_image).shape == torch.Size((4,i+1,112,112))