
import logging

from typing import Union
from pathlib import Path
from collections import namedtuple

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import cv2 as cv2

from torch_ihc_segmentation.data import augmentation

SegmentationPair = namedtuple("SegmentationPair", "img, mask")

class PairedDirDataset(Dataset):
    def __init__(self, basedir: Union[str, Path], augment=False) -> None:
        super().__init__()
        
        img_path = Path(basedir) / "img"
        mask_path = Path(basedir) / "mask"

        self._img_path = img_path
        self._mask_path = mask_path

        assert img_path.is_dir()
        assert mask_path.is_dir()

        self._base_img_names = tuple([x.name for x in img_path.glob("*.png")])
        logging.info(f"Found {len(self._base_img_names)} images in the {str(img_path)} directory")

        for x in self._base_img_names:
            assert (img_path/x).is_file()
            assert (mask_path/x).is_file()

        self._aug = augment

    @property
    def augment(self):
        return self._aug

    def set_augment(self, aug:bool):
        logging.info("Setting augmentation to ", aug)
        self._aug = aug

    def __getitem__(self, index):
        x = self._base_img_names[index]
        # Read an image with OpenCV and convert it to the RGB colorspace
        img = cv2.imread(str(self._img_path / x), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self._mask_path / x), cv2.IMREAD_GRAYSCALE)

        transformed = augmentation.preaug_transform(image=img, mask=mask)

        # Augment an image
        if self._aug:
            transformed = augmentation.aug_transform(image=transformed["img"], mask=transformed["mask"])

        transformed = augmentation.postaug_transform(image=transformed["img"], mask=transformed["mask"])

        return SegmentationPair(transformed['img'], transformed['mask'])

    def __len__(self):
        return len(self._base_img_names)

