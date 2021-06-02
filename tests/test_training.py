
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch_ihc_segmentation as TIS

model = TIS.models.LitSegmentationModel(pretrained=False)
model.steps_per_epoch = 4
trainer = pl.Trainer(fast_dev_run = True)

sample_image = torch.rand([4,3,224,224])
sample_gt = torch.randint(high = 4, size = (4,224,224))

sample_dataloader = DataLoader(TensorDataset(sample_image, sample_gt))

trainer.fit(model, sample_dataloader)