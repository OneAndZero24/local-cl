import os
import numpy as np
from typing import Dict
from PIL import Image
from continuum.datasets.base import _ContinuumDataset
from continuum.scenarios import ClassIncremental


class ImageNetSubset10(_ContinuumDataset):
    def __init__(self, data_path: str, class_names_to_wnid: dict, train: bool = True, image_size: int = 224):
        self.data_path = os.path.join(data_path, "train" if train else "test")
        self.class_names_to_wnid = class_names_to_wnid
        self.class_name_to_label = {name: i for i, name in enumerate(class_names_to_wnid.keys())}
        self.image_size = image_size
        self._x, self._y, self._t = None, None, None
        self._bboxes = None

    def _setup(self):
        """Load all selected classes as numpy image arrays."""
        all_x, all_y = [], []
        print(f"Loading subset from: {self.data_path}")

        for class_name, wnid in self.class_names_to_wnid.items():
            label = self.class_name_to_label[class_name]
            folder = os.path.join(self.data_path, wnid)

            if not os.path.isdir(folder):
                print(f"Missing folder for '{class_name}' ({wnid})")
                continue

            img_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for path in img_files:
                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize((self.image_size, self.image_size))
                    all_x.append(np.array(img, dtype=np.uint8))
                    all_y.append(label)
                except Exception as e:
                    print(f"Could not load {path}: {e}")

        self._x, self._y, self._t = np.array(all_x), np.array(all_y), np.zeros(len(all_x))
        print(f"Loaded {len(all_x)} images across {len(set(all_y))} classes.")


    def get_data(self):
        if self._x is None:
            self._setup()
        return self._x, self._y, self._t

    def get_sample(self, index: int):
        """Load and transform one image (path → tensor)."""
        path = self._x[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self._y[index]
        task = self._t[index]
        return img, label, task


def build_imagenet_subset10_scenario(data_path: str, class_names_to_wnid: Dict[str, str]):
    dataset = ImageNetSubset10(data_path=data_path, class_names_to_wnid=class_names_to_wnid, train=True)
    scenario = ClassIncremental(dataset, increment=2)
    return scenario
