import time
import math

from evaluator.f1_evaluator import F1Evaluator
from figo.affinity_sampler import AffinitySampler
from loader.cache_res_loader import CacheResLoader
from modeling.efficientdet.model_wrapper import EfficientDetModelWrapper
from figo.estimator import Estimator
from figo.chunk import Chunk
from predicate.count_predicate import CountPredicate
from sampler.uniform_sampler import UniformSampler
from profiler.model_time import lookup_model_time


class Scheduler:
    def __init__(
        self,
        modeling_name,
        img_loader,
        pred,
        use_cache=False,
        accuracy=0.95,
        generalization_error=0.03,
    ):
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

        self._accuracy = accuracy
        self._generalization_error = generalization_error

        self._esti = Estimator(self._model_ensemble[:-1])

        self._name = "FiGO scheduler"

        self._query_time = 0
        self._optimization_time = 0

        self._existing_sample_idx = [
            [] for _ in range(len(self._model_ensemble))
        ]
        self._existing_sample_res = [
            dict() for _ in range(len(self._model_ensemble))
        ]

        self._accuracy_list = [[] for _ in range(len(self._model_ensemble))]

        self._plan_chunk = []

        self._exec_record = []

        self._res = dict()

    def _log(self, msg):
        print(self._name, msg)

    def _benefit(self, start, end, used_model, sample_best_model_record):
        opt_plus = 10 * sum(
            [lookup_model_time(self._model_ensemble[m]) for m in used_model]
        )
        cur_exec_t = (end - start) * lookup_model_time(
            self._model_ensemble[max(sample_best_model_record)]
        )

        left_sample_best_model_record = sample_best_model_record[
            : len(sample_best_model_record) // 2
        ]
        left_best_model = self._model_ensemble[
            max(left_sample_best_model_record)
        ]
        next_left_exec_t = (
            (end - start) / 2 * lookup_model_time(left_best_model)
        )

        right_sample_best_model_record = sample_best_model_record[
            len(sample_best_model_record) // 2 :
        ]
        right_best_model = self._model_ensemble[
            max(right_sample_best_model_record)
        ]
        next_right_exec_t = (
            (end - start) / 2 * lookup_model_time(right_best_model)
        )

        next_exec_t = next_left_exec_t + next_right_exec_t

        exec_minus = cur_exec_t - next_exec_t

        return exec_minus - opt_plus

    def _sample_size_lower_bound(self, accuracy_list, num_model):
        mean_acc = sum(accuracy_list) / len(accuracy_list)
        dev_acc = sum([abs(acc - mean_acc) for acc in accuracy_list]) / len(
            accuracy_list
        )
        var_acc = dev_acc**2

        # overcome division by zero --> the observed accuracy is good
        # so we project very few samples needed
        if self._accuracy + self._generalization_error - mean_acc == 0:
            return 1

        k = (
            math.log(num_model, math.e)
            * (
                4 * var_acc
                + (2 / 3)
                * (self._accuracy + self._generalization_error - mean_acc)
            )
            / (self._accuracy + self._generalization_error - mean_acc) ** 2
        )

        return k

    def _traverse(self, depth, start, end):
        if depth == 1:
            used_model = list(range(len(self._model_ensemble)))
        else:
            used_model = self._esti.get_best_n_model(n=2) + [
                len(self._model_ensemble) - 1
            ]
            used_model = sorted(used_model)

        # affinity sampling based on top of uniform sampling
        current_sample_idx = AffinitySampler().get_idx(
            start, end, self._existing_sample_idx
        )

        sample_best_model_record = []

        for s_idx in current_sample_idx:
            best_model = len(self._model_ensemble) - 1

            ref_out = None

            for m_idx in range(len(self._model_ensemble) - 1, -1, -1):

                # actual pruning
                if m_idx not in used_model:
                    continue

                # if no sampling record, inference
                if s_idx not in self._existing_sample_idx[m_idx]:
                    if not self._use_cache:
                        img_path = self._loader.get_img_path(s_idx)
                        out = self._model_ensemble[m_idx].predict(img_path)
                    else:
                        out = self._model_ensemble[m_idx].cached_predict(s_idx)
                        self._exec_record.append(self._model_ensemble[m_idx])

                    # insert into cache for later use
                    self._existing_sample_res[m_idx][s_idx] = out
                    self._existing_sample_idx[m_idx].append(s_idx)
                else:
                    out = self._existing_sample_res[m_idx][s_idx]

                if m_idx == len(self._model_ensemble) - 1:
                    ref_out = out
                    self._res[s_idx] = out

                # update model estimator
                if m_idx != len(self._model_ensemble) - 1:
                    reward = self._pred.evaluate(out) == self._pred.evaluate(
                        ref_out
                    )
                    self._esti.update_r(m_idx, reward)

                if self._pred.evaluate(out) == self._pred.evaluate(ref_out):
                    best_model = m_idx
                    self._accuracy_list[m_idx].append(1)
                else:
                    for k in range(m_idx, -1, -1):
                        self._accuracy_list[k].append(0)
                    break

            if not self._pred.evaluate(ref_out):
                best_model = -1

            sample_best_model_record.append(best_model)

        current_accuracy_list = []
        for m_idx in used_model:
            current_accuracy_list += self._accuracy_list[m_idx]

        if (
            self._sample_size_lower_bound(
                current_accuracy_list, num_model=len(used_model)
            )
            < 10
            and self._benefit(start, end, used_model, sample_best_model_record)
            < 0
        ) or (end - start) < 100:
            sel_model = max(sample_best_model_record)

            chunk = Chunk()
            chunk.set_range(start, end)

            if sel_model != -1:
                chunk.set_model(self._model_ensemble[sel_model])
            else:
                chunk.set_model(None)

            self._plan_chunk.append(chunk)
        else:
            span = (end - start) // 2
            self._traverse(depth + 1, start, start + span)
            self._traverse(depth + 1, start + span, end)

    def _optimize(self):
        st = time.perf_counter()
        self._traverse(1, 0, len(self._loader))

        if not self._use_cache:
            self._optimization_time = time.perf_counter() - st
        else:
            for m in self._exec_record:
                self._optimization_time += lookup_model_time(m)

    def _execute(self):
        for chunk in self._plan_chunk:
            for i in range(*chunk.range):
                model = chunk.model

                if i in self._res:
                    continue

                if model is None:
                    out = {"class": [], "score": []}
                else:
                    if not self._use_cache:
                        img_path = self._loader.get_img_path(i)
                        out = model.predict(img_path)
                    else:
                        out = model.cached_predict(i)
                        self._exec_record.append(model)

                self._res[i] = out

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
