import os
import json


class CacheResLoader:
    def __init__(self, dataset, model_name):
        with open(
            os.path.join("./cache", dataset, "{}.json".format(model_name))
        ) as f:
            temp_res_dict = json.load(f)

        # modify res_dict
        self._res_dict = dict()
        for k, res in temp_res_dict.items():
            self._res_dict[int(k)] = res

    def load(self, idx):
        return self._res_dict[idx]
