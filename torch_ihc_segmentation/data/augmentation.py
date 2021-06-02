
import albumentations as A
import cv2

preaug_transform = A.Compose([
    A.RandomCrop(width=600, height=600),
])

# Declare an augmentation pipeline
aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


postaug_transform = A.Compose([
    A.RandomCrop(width=600, height=600),
])