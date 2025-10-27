import time

from loader.cache_res_loader import CacheResLoader
from modeling.efficientdet.model_wrapper import EfficientDetModelWrapper
from profiler.model_time import lookup_model_time


class Scheduler:
    def __init__(self, modeling_name, img_loader, pred, use_cache=False):
        if modeling_name == "efficientdet":
            self._model = EfficientDetModelWrapper(7, use_cache)

            if use_cache:
                cache_res = CacheResLoader(
                    img_loader.get_dataset_name(), self._model.get_name()
                )
                self._model.load_cache_res(cache_res)

        self._loader = img_loader
        self._pred = pred
        self._use_cache = use_cache

        self._name = "Naive scheduler"

        self._query_time = 0

        self._exec_record = []

        self._res = dict()

    def _log(self, msg):
        print(self._name, msg)

    def _optimize(self):
        pass

    def _execute(self):
        for i in range(len(self._loader)):

            # self._log("process img {}".format(i))

            if not self._use_cache:
                img_path = self._loader.get_img_path(i)
                self._res[i] = self._model.predict(img_path)
            else:
                self._exec_record.append(self._model)
                self._res[i] = self._model.cached_predict(i)

    def process(self):
        st = time.perf_counter()

        self._log("start optimization ...")
        self._optimize()

        self._log("start execution ...")
        self._execute()

        if not self._use_cache:
            self._query_time = time.perf_counter() - st
        else:
            for m in self._exec_record:
                self._query_time += lookup_model_time(m)

        return self._res

    def get_query_time(self):
        return self._query_time

    def get_optimization_time(self):
        return 0

    def get_execution_time(self):
        return self._query_time
