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
            self._model_ensemble = [
                EfficientDetModelWrapper(i, use_cache) for i in range(8)
            ]

            if use_cache:
                for i in range(8):
                    model_cache_res = CacheResLoader(
                        img_loader.get_dataset_name(),
                        self._model_ensemble[i].get_name(),
                    )
                    self._model_ensemble[i].load_cache_res(model_cache_res)

        self._loader = img_loader
        self._pred = pred
        self._use_cache = use_cache

        self._name = "ME-Coarse scheduler"

        self._query_time = 0
        self._optimization_time = 0

        self._model_idx = -1

        self._exec_record = []

        self._res = dict()

    def _log(self, msg):
        print(self._name, msg)

    def _optimize(self):
        sampler = UniformSampler(self._loader)
        sample_idx = sampler.get_idx()

        model_ensemble_res = [dict() for _ in range(len(self._model_ensemble))]

        st = time.perf_counter()

        # run reference on all model ensembles
        for i in range(len(self._model_ensemble)):
            for s_idx in sample_idx:
                if not self._use_cache:
                    img_path = self._loader.get_img_path(s_idx)
                    out = self._model_ensemble[i].predict(img_path)
                else:
                    self._exec_record.append(self._model_ensemble[i])
                    out = self._model_ensemble[i].cached_predict(s_idx)

                model_ensemble_res[i][s_idx] = out

        # evaluate all models in ensemble and pick the best model
        ref_res = model_ensemble_res[-1]

        for k, res in ref_res.items():
            self._res[k] = res

        best_plan_list = []

        for s_idx in sample_idx:

            if self._pred.evaluate(ref_res[s_idx]):
                best_plan = len(self._model_ensemble) - 1

                for i in range(len(self._model_ensemble) - 1, -1, -1):
                    if self._pred.evaluate(
                        model_ensemble_res[i][s_idx]
                    ) and self._pred.evaluate(ref_res[s_idx]):
                        best_plan = i
                    else:
                        break
            else:
                best_plan = -1

            best_plan_list.append(best_plan)

        self._model_idx = max(best_plan_list)

        if not self._use_cache:
            self._optimization_time = time.perf_counter() - st
        else:
            for m in self._exec_record:
                self._optimization_time += lookup_model_time(m)

    def _execute(self):
        for i in range(len(self._loader)):

            # self._log("process {}".format(i))

            if i in self._res:
                continue

            if self._model_idx == -1:
                self._res[i] = {"class": [], "score": []}

            if not self._use_cache:
                img_path = self._loader.get_img_path(i)
                self._res[i] = self._model_ensemble[self._model_idx].predict(
                    img_path
                )
            else:
                self._exec_record.append(self._model_ensemble[self._model_idx])
                self._res[i] = self._model_ensemble[
                    self._model_idx
                ].cached_predict(i)

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
