import os
import json


class ImageLoader:
    def __init__(self, dataset, use_cache=False):
        self._dataset = dataset
        self._use_cache = use_cache

        if not use_cache:
            self._root_dir = os.path.join("./data", dataset)
            self._length = len(os.listdir(self._root_dir))

            self._img_path_list = []

            if self._dataset == "ua-detrac":
                self._img_path_list = os.listdir(self._root_dir)
        else:
            with open(
                os.path.join("./cache", dataset, "efficientdet-d7.json")
            ) as f:
                res = json.load(f)
            self._length = len(res)

    def __len__(self):
        return self._length

    def get_img_path(self, idx):
        if idx >= self._length:
            raise Exception(
                "Dataset {} index is out of range".format(self._dataset)
            )

        if self._dataset == "ua-detrac":
            img_path = os.path.join(self._root_dir, self._img_path_list[idx])
        else:
            img_path = "frame" + str(idx) + ".jpg"
            img_path = os.path.join(self._root_dir, img_path)

        return img_path

    def get_dataset_name(self):
        return self._dataset
