try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # Python <3.8 compatibility
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

import torch_ihc_segmentation.lovasz_losses as lovasz_losses
import torch_ihc_segmentation.losses as losses
import torch_ihc_segmentation.models as models
