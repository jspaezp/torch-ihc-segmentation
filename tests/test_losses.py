
import torch
import torch.nn.functional as F
import torch_ihc_segmentation as TIS

sample_image = torch.rand([4,3,224,224])
sample_gt = torch.randint(high = 4, size = (4,224,224))
sample_out_logits = (torch.rand([4,5,224,224]) - 0.5) * 50

sample_out_probs = F.softmax(sample_out_logits, dim = 1)
print(sample_out_probs[0,...,0,0].sum())

lovasz_out = TIS.lovasz_losses.lovasz_softmax(probas = sample_out_probs, labels = sample_gt)
print(lovasz_out)