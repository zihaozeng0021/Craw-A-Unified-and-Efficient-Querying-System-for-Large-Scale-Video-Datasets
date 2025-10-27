from audioop import avg
import time


model_time_dict = dict()


def lookup_model_time(model):
    name = model.get_name()

    if name not in model_time_dict:
        # warmup
        model.predict("./data/ua-detrac/img00001.jpg")

        st = time.perf_counter()

        for _ in range(10):
            out = model.predict("./data/ua-detrac/img00001.jpg")

        avg_lat = (time.perf_counter() - st) / 10
        model_time_dict[name] = avg_lat

    return model_time_dict[name]
