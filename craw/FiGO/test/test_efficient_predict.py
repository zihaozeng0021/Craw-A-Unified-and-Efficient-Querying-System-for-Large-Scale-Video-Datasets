import time
from modeling.efficientdet.model_label import cat_id_to_label

import setup_module

from modeling.efficientdet.model_wrapper import EfficientDetModelWrapper
from loader.image_loader import ImageLoader


def main():
    model = EfficientDetModelWrapper(idx=0)

    loader = ImageLoader("ua-detrac")

    print("Start prediction ... ")

    for i in range(len(loader)):
        img_path = loader.get_img_path(i)

        st = time.perf_counter()
        out = model.predict(img_path)

        print(out)
        print("{:.2f} sec".format(time.perf_counter() - st))


if __name__ == "__main__":
    main()
