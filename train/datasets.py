import json
import os
import random
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import webdataset as wds


class COCOFlickrDataset(Dataset):
    def __init__(
            self,
            image_dir_path,
            annotations_path,
            transform=None,
            is_flickr=False,
            prefix=None,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr
        self.transform = transform
        self.prefix = prefix

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.prefix}{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return self.transform(image), caption


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        # target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return sample, target


class COCOCaptionDataset(Dataset):
    """COCO Captions dataset for CLIP alignment training.

    Indexes by image and randomly picks one caption per sample.
    """

    def __init__(self, image_dir, annotation_file, transform, tokenizer):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        annotations = json.load(Path(annotation_file).open("rt"))

        self.img_id_to_filename = {}
        for img_info in annotations["images"]:
            self.img_id_to_filename[img_info["id"]] = img_info["file_name"]

        self.img_id_to_captions = {}
        for ann in annotations["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_captions:
                self.img_id_to_captions[img_id] = []
            self.img_id_to_captions[img_id].append(ann["caption"])

        self.img_ids = list(self.img_id_to_filename.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        caption = random.choice(self.img_id_to_captions[img_id])

        img_path = os.path.join(self.image_dir, self.img_id_to_filename[img_id])
        image = Image.open(img_path).convert("RGB")

        img_tensor = self.transform(image)
        text_tensor = self.tokenizer(caption)[0]  # squeeze batch dim: (1, seq_len) -> (seq_len,)
        return img_tensor, text_tensor


# def make_wds_dataset(shards_path, transform, batch_size, steps_per_gpu, tokenizer):
#     dataset = (
#         wds.WebDataset(
#             shards_path,
#             shardshuffle=True,
#             nodesplitter=wds.split_by_node,
#             resampled=True,
#         )
#         .shuffle(1000)
#         .decode("pil")
#         .rename(image="jpg;png;jpeg;webp", text="txt")
#         .map_dict(
#             image=transform,
#             text=lambda t: tokenizer(t)[0]  # [0] 去掉 batch 维，shape: (seq_len,)
#         )
#         .to_tuple("image", "text")
#         .batched(batch_size, partial=False)
#         .with_epoch(steps_per_gpu)
#     )
#     return dataset


def make_wds_dataset(shards_path, transform, batch_size, steps_per_gpu, tokenizer):

    def is_valid_sample(sample):
        """过滤缺失字段的样本"""
        return "jpg" in sample or "png" in sample or "jpeg" in sample or "webp" in sample

    dataset = (
        wds.WebDataset(
            shards_path,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            resampled=True,
            handler=wds.warn_and_continue,      # ① shard 级别出错跳过
        )
        .shuffle(1000)
        .select(is_valid_sample)                # ② 过滤没有图片字段的样本
        .decode(
            "pil",
            handler=wds.warn_and_continue       # ③ 截断/损坏图片跳过
        )
        .rename(
            image="jpg;png;jpeg;webp",
            text="txt",
            handler=wds.warn_and_continue       # ④ 字段缺失跳过
        )
        .map_dict(
            image=transform,
            text=lambda t: tokenizer(t)[0],
            handler=wds.warn_and_continue       # ⑤ transform/tokenize 异常跳过
        )
        .to_tuple("image", "text")
        .batched(batch_size, partial=False)
        .with_epoch(steps_per_gpu)              # ⑥ 循环读取，保证每epoch有足够样本
    )
    return dataset