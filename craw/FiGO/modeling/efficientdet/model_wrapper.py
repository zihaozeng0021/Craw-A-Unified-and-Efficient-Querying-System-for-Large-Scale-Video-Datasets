import setup_module
import os
import torch

from backbone import EfficientDetBackbone

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (
    preprocess,
    invert_affine,
    postprocess,
)

from modeling.efficientdet.model_label import (
    cat_id_to_label,
    label_list,
    out_to_std_out,
)
from modeling.abc import ModelWrapper

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2


class EfficientDetModelWrapper(ModelWrapper):
    def __init__(self, idx, use_cache=False):
        self._idx = idx
        self._use_cache = use_cache

        self._cache_res = None
        self._name = "efficientdet-d{}".format(idx)

        if not use_cache:
            self._model = EfficientDetBackbone(
                compound_coef=idx,
                num_classes=len(label_list),
                ratios=anchor_ratios,
                scales=anchor_scales,
            )

            self._use_cuda = torch.cuda.is_available()

            self._load_weight(
                os.path.join(
                    "./weights/efficientdet/efficientdet-d{}.pth".format(idx)
                )
            )

    def _load_weight(self, weight_path):
        self._model.load_state_dict(
            torch.load(weight_path, map_location="cpu")
        )
        self._model.requires_grad_(False)
        self._model.eval()

        if self._use_cuda:
            self._model.cuda()

    def load_cache_res(self, cache_res):
        self._cache_res = cache_res

    def predict(self, img_path):
        ori_imgs, framed_imgs, framed_metas = preprocess(
            img_path, max_size=input_sizes[self._idx]
        )

        if self._use_cuda:
            x = torch.stack(
                [torch.from_numpy(fi).cuda() for fi in framed_imgs], 0
            )
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = self._model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(
                x,
                anchors,
                regression,
                classification,
                regressBoxes,
                clipBoxes,
                threshold,
                iou_threshold,
            )

        out = invert_affine(framed_metas, out)

        out = out_to_std_out(out)

        for k in range(len(out["class"])):
            out["class"][k] = cat_id_to_label(out["class"][k])

        return out

    def cached_predict(self, idx):
        out = self._cache_res.load(idx)

        for k in range(len(out["class"])):
            out["class"][k] = cat_id_to_label(out["class"][k])

        return out

    def get_name(self):
        return self._name
