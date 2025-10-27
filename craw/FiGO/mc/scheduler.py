import time
from tkinter import W

from evaluator.f1_evaluator import F1Evaluator
from loader.cache_res_loader import CacheResLoader
from modeling.efficientdet.model_wrapper import EfficientDetModelWrapper
from predicate.count_predicate import CountPredicate
from sampler.uniform_sampler import UniformSampler
from profiler.model_time import lookup_model_time


def gen_conf_seq():
    for a in range(9, 4, -1):
        for b in range(a, 4, -1):
            for c in range(b, 4, -1):
                for d in range(c, 4, -1):
                    for e in range(d, 4, -1):
                        for f in range(e, 4, -1):
                            for g in range(f, 4, -1):
                                yield (
                                    a / 10,
                                    b / 10,
                                    c / 10,
                                    d / 10,
                                    e / 10,
                                    f / 10,
                                    g / 10,
                                )


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

        self._name = "MC scheduler"

        self._query_time = 0
        self._optimization_time = 0

        self._model_conf = None

        self._exec_record = []

        self._res = dict()

    def _log(self, msg):
        print(self._name, msg)

    def _inference(self, conf_seq, res_list):
        tmp_count, ret_res = [0 for _ in range(len(self._pred.count))], None

        stop_m_idx = len(self._model_ensemble) - 1

        plan = []

        def _greater(actual, expected):
            for i, r in enumerate(actual):
                if r < expected[i]:
                    return False
            return True

        for i, res in enumerate(res_list):
            for k, c in enumerate(res["class"]):
                for c_idx in range(len(self._pred.cls)):
                    if (
                        i != len(self._model_ensemble) - 1
                        and c == self._pred.cls[c_idx]
                        and res["score"][k] >= conf_seq[i]
                    ):
                        tmp_count[c_idx] += 1

            plan.append(self._model_ensemble[i])
            ret_res = res

            if _greater(tmp_count, self._pred.count):
                stop_m_idx = i
                break

        ref_res = res_list[len(self._model_ensemble) - 1]
        ref_count = [0 for _ in range(len(self._pred.cls))]

        for k, c in enumerate(ref_res):
            for c_idx in range(len(self._pred.cls)):
                if c == self._pred.cls[c_idx] and ref_res["score"][k] >= 0.5:
                    ref_count[c_idx] += 1

        inf_time = 0
        for i in range(stop_m_idx + 1):
            inf_time += lookup_model_time(self._model_ensemble[i])
        return (
            ret_res,
            _greater(tmp_count, self._pred.count)
            == _greater(ref_count, self._pred.count),
            inf_time,
            plan,
        )

    def _optimize(self):
        sampler = UniformSampler(self._loader)
        sample_idx = sampler.get_idx()

        res_list = []

        st = time.perf_counter()

        for s_idx in sample_idx:
            sample_res_list = []

            for _, m in enumerate(self._model_ensemble):
                if not self._use_cache:
                    img_path = self._loader.get_img_path(s_idx)
                    out = m.predict(img_path)
                else:
                    self._exec_record.append(m)
                    out = m.cached_predict(s_idx)

                sample_res_list.append(out)

            res_list.append(sample_res_list)

            # cache reference results
            self._res[s_idx] = sample_res_list[len(self._model_ensemble) - 1]

        conf_seq_stats = dict()

        for conf_seq in gen_conf_seq():
            stats = dict()
            stats["correct"] = 0
            stats["inf-time"] = 0

            for sample_res_list in res_list:
                _, correct, inf_time, _ = self._inference(
                    conf_seq, sample_res_list
                )
                stats["correct"] += correct
                stats["inf-time"] += inf_time

            conf_seq_stats[conf_seq] = stats

        inf_time = 0xFFFFFFFF

        # only optimize inference time because higher
        # accuracy model requires even longer time, which
        # is not comparable to other approaches
        for conf_seq, stats in conf_seq_stats.items():
            if stats["inf-time"] < inf_time:
                self._model_conf = conf_seq
                inf_time = stats["inf-time"]

        if not self._use_cache:
            self._optimization_time = time.perf_counter() - st
        else:
            for m in self._exec_record:
                self._optimization_time += lookup_model_time(m)

    def _execute(self):
        for i in range(len(self._loader)):
            if i in self._res:
                continue

            if not self._use_cache:
                raise Exception(
                    "Execution with no cache is not currently supported."
                )
            else:
                res_list = []
                for m in self._model_ensemble:
                    out = m.cached_predict(i)
                    res_list.append(out)

                res, _, _, exec_plan = self._inference(
                    self._model_conf, res_list
                )

                self._res[i] = res
                self._exec_record += exec_plan

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
