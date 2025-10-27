import time

from evaluator.f1_evaluator import F1Evaluator
from loader.cache_res_loader import CacheResLoader
from modeling.efficientdet.model_wrapper import EfficientDetModelWrapper
from predicate.count_predicate import CountPredicate
from sampler.uniform_sampler import UniformSampler
from profiler.model_time import lookup_model_time


class Scheduler:
    def __init__(self, modeling_name, img_loader, pred, use_cache=False):
        if modeling_name == "efficientdet":
            self._ref_model = EfficientDetModelWrapper(7, use_cache)

            self._filter_model_list = [
                EfficientDetModelWrapper(i, use_cache) for i in range(7)
            ]

            if use_cache:
                ref_cache_res = CacheResLoader(
                    img_loader.get_dataset_name(), self._ref_model.get_name()
                )
                self._ref_model.load_cache_res(ref_cache_res)

                for i in range(7):
                    filter_cache_res = CacheResLoader(
                        img_loader.get_dataset_name(),
                        self._filter_model_list[i].get_name(),
                    )
                    self._filter_model_list[i].load_cache_res(filter_cache_res)

        self._loader = img_loader
        self._pred = pred
        self._use_cache = use_cache

        self._name = "MS-Filter scheduler"

        self._query_time = 0
        self._optimization_time = 0

        self._filter_idx = -1

        self._exec_record = []

        self._res = dict()

    def _log(self, msg):
        print(self._name, msg)

    def _optimize(self):
        sampler = UniformSampler(self._loader)
        sample_idx = sampler.get_idx()

        ref_res = dict()

        st = time.perf_counter()

        # run reference model first and only once
        for s_idx in sample_idx:
            if not self._use_cache:
                img_path = self._loader.get_img_path(s_idx)
                out = self._ref_model.predict(img_path)
            else:
                self._exec_record.append(self._ref_model)
                out = self._ref_model.cached_predict(s_idx)
            ref_res[s_idx] = out

            # since refenrece model already inferences,
            # append to final results to save model invocation
            self._res[s_idx] = out

        # inference with different filters onward
        binary_pred = CountPredicate(
            self._pred.cls, [1 for _ in range(len(self._pred.cls))]
        )

        filter_res_list = [dict() for _ in range(len(self._filter_model_list))]

        for i in range(len(self._filter_model_list)):
            for s_idx in sample_idx:
                if not self._use_cache:
                    img_path = self._loader.get_img_path(s_idx)
                    out = self._filter_model_list[i].predict(img_path)
                else:
                    self._exec_record.append(self._filter_model_list[i])
                    out = self._filter_model_list[i].cached_predict(s_idx)

                if binary_pred.evaluate(out):
                    filter_res_list[i][s_idx] = ref_res[s_idx]
                else:
                    filter_res_list[i][s_idx] = {"class": [], "score": []}

        # evaluate every setup
        evaluator = F1Evaluator(self._pred)

        for i in range(len(self._filter_model_list)):
            f1 = evaluator.evaluate(filter_res_list[i], ref_res)
            self._filter_idx = i
            if f1 >= 0.95:
                break

        if not self._use_cache:
            self._optimization_time = time.perf_counter() - st
        else:
            for m in self._exec_record:
                self._optimization_time += lookup_model_time(m)

    def _execute(self):
        binary_pred = CountPredicate(
            self._pred.cls, [1 for _ in range(len(self._pred.cls))]
        )

        for i in range(len(self._loader)):

            # self._log("process {}".format(i))

            if i in self._res:
                continue

            if not self._use_cache:
                img_path = self._loader.get_img_path(i)

                filter_out = self._filter_model_list[self._filter_idx].predict(
                    img_path
                )

                if binary_pred.evaluate(filter_out):
                    self._res[i] = self._ref_model.predict(img_path)
                else:
                    self._res[i] = {"class": [], "score": []}
            else:
                self._exec_record.append(
                    self._filter_model_list[self._filter_idx]
                )

                filter_out = self._filter_model_list[
                    self._filter_idx
                ].cached_predict(i)

                if binary_pred.evaluate(filter_out):
                    self._exec_record.append(self._ref_model)
                    self._res[i] = self._ref_model.cached_predict(i)
                else:
                    self._res[i] = {"class": [], "score": []}

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

    def get_execution_time(self):
        return self._query_time - self._optimization_time

    def get_optimization_time(self):
        return self._optimization_time
