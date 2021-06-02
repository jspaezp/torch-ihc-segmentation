import logging
from typing import Any, Optional, Dict
from collections import namedtuple
import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import optim
import torchvision.models.segmentation as torchvision_models
import pytorch_lightning as pl

import torch_ihc_segmentation.lovasz_losses as lls

DefaultArgsFunPair = namedtuple("DefaultArgsFunPair", "fun, args")


class LitSegmentationModel(pl.LightningModule):
    _available_optimizers = {
        "Adam": DefaultArgsFunPair(
            optim.AdamW,
            {
                "lr": 3e-4,
            },
        ),
    }
    _available_schedulers = {
        "OneCycle": DefaultArgsFunPair(
            optim.lr_scheduler.OneCycleLR,
            {
                "max_lr": 3e-4,
                "pct_start": 0.1,
                "div_factor": 25,
                "final_div_factor": 1e6,
                "three_phase": True,
                "total_steps": None,
                "epochs": None,
                "steps_per_epoch": None,
            },
        ),
    }

    def __init__(
        self, pretrained=True, num_classes=5, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = torchvision_models.fcn_resnet50(
            pretrained=pretrained, num_classes=num_classes
        )
        self.losses = torch.nn.ModuleDict(
            {
                "lovasz": lls.LovaszSoftmax(),
            }
        )

    def forward(self, *args, **kwargs) -> Any:
        return self.model.forward(*args, **kwargs)["out"]

    def configure_optimizers(
        self,
        optimizer="Adam",
        scheduler="OneCycle",
        optim_args_dict={},
        sched_args_dict={},
    ):
        # Get non frozen parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Setting up optimizer
        optimizer_pair = self._available_optimizers[optimizer]
        optim_args_dict = copy.deepcopy(optim_args_dict)
        optim_args_dict.update(optimizer_pair.args)
        optimizer = optimizer_pair.fun(params=params, **optim_args_dict)

        # Setting up scheduler
        scheduler_pair = self._available_schedulers[scheduler]

        spe = self.steps_per_epoch // self.trainer.accumulate_grad_batches
        if spe is None:
            logging.warning("Asuming 100 as the number of steps per epoch ...")

        sch_args = copy.deepcopy(scheduler_pair.args)
        sch_args.update(sched_args_dict)
        update_args = {"epochs": self.trainer.max_epochs or 10, "steps_per_epoch": spe}
        update_args = {k: v for k, v in update_args.items() if k in sch_args}
        sch_args.update(update_args)

        scheduler_dict = {
            "scheduler": scheduler_pair.fun(optimizer=optimizer, **sch_args),
            "interval": "step",
        }

        return [optimizer], [scheduler_dict]

    def _step(self, batch, batch_idx=None):
        patches, classes = batch[0], batch[1]

        logits = self.forward(patches)
        yhat = torch.argmax(logits, dim=1)

        losses = {k: v(logits, classes) for k, v in self.losses.items()}
        total_loss = 0
        for _, v in losses.items():
            total_loss += v

        out = {
            "l": total_loss,
        }

        return out

    def training_step(
        self, batch, batch_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """See pytorch_lightning documentation."""
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"t_" + k: v for k, v in step_out.items()}
        log_dict.update({"LR": self.trainer.optimizers[0].param_groups[0]["lr"]})

        self.log_dict(
            log_dict,
            prog_bar=True,
        )

        return {"loss": step_out["l"]}

    def validation_step(self, batch, batch_idx: Optional[int] = None) -> None:
        """See pytorch_lightning documentation."""
        step_out = self._step(batch, batch_idx=batch_idx)
        log_dict = {"v_" + k: v for k, v in step_out.items()}

        self.log_dict(
            log_dict,
            prog_bar=True,
        )
