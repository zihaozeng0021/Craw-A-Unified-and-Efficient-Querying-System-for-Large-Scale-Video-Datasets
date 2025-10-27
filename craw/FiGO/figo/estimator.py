"""
Module for pruning models.
"""
from functools import cmp_to_key
import numpy as np

from profiler.model_time import lookup_model_time
from modeling.abc import ModelWrapper


class ModelEstimator:
    def __init__(self):
        self._tau = 0.0001
        self._phi = 1

        self._n = 0
        self._Q = 0

    def update(self, r):
        self._n += 1

        self._Q = (1 - 1 / self._n) * self._Q + (1 / self._n) * r

        self._phi = ((self._tau * self._phi) + (self._n * self._Q)) / (
            self._tau + self._n
        )

        self._tau += 1

    def get_estimate_r(self):
        return np.random.rand() / np.sqrt(self._tau) + self._phi


class Estimator:
    def __init__(self, model_ensemble):
        self._model_ensemble = model_ensemble

        self._estimator_list = [
            ModelEstimator() for _ in range(len(self._model_ensemble))
        ]

        self._cost_list = [lookup_model_time(m) for m in model_ensemble]

    def update_r(self, m, r):
        if isinstance(m, ModelWrapper):
            m_idx = self._model_ensemble.index(m)
            self._estimator_list[m_idx].update(r)
        else:
            self._estimator_list[m].update(r)

    def get_best_n_model(self, n):
        score_list = []

        for i, esti in enumerate(self._estimator_list):
            score_list.append((esti.get_estimate_r(), self._cost_list[i], i))

        def _cmp_model(m1, m2):
            if m1[0] != m2[0]:
                return -(m1[0] - m2[0])
            else:
                return -(m1[1] - m2[1])

        sorted_score_list = sorted(score_list, key=cmp_to_key(_cmp_model))
        return [sorted_score_list[i][2] for i in range(n)]
