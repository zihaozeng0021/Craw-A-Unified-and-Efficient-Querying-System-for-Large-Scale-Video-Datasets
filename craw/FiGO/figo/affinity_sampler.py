class AffinitySampler:
    def __init__(self, ratio=0.1):
        self._ratio = ratio

    def get_idx(self, start, end, existing_sample_idx):
        """
        Parameters
        ----------
        existing_sample_idx : List[List[]]
            A 2-D list contains already sampled image index. The
            first dimension contains all models specified by the
            scheduler. The second dimension contains all sampled
            indices of that model.
        """
        span = int((end - start) * self._ratio)

        # original expected sampled indices
        target_sample_idx = list(range(start, end, span))

        inrange_existing_sample_idx = []
        for i in range(len(existing_sample_idx)):
            for k in existing_sample_idx[i]:
                if start <= k < end and k not in inrange_existing_sample_idx:
                    inrange_existing_sample_idx.append(k)

        # generate actual sample indices
        actual_sample_idx = []

        if len(inrange_existing_sample_idx) == 0:
            actual_sample_idx = target_sample_idx
        elif 0 < len(inrange_existing_sample_idx) < 1 / self._ratio:
            actual_sample_idx += inrange_existing_sample_idx

            for k in inrange_existing_sample_idx:
                best_dis, best_idx = -1, -1

                for idx in target_sample_idx:
                    if (
                        abs(idx - k) > best_dis
                        and idx not in existing_sample_idx
                    ):
                        best_idx = idx
                        best_dis = abs(idx - k)

                actual_sample_idx.append(best_idx)
                target_sample_idx.remove(best_idx)

                if len(actual_sample_idx) == 1 / self._ratio:
                    break
        else:
            for idx in target_sample_idx:
                best_dis, best_idx = 0xFFFFFFFF, -1

                for k in inrange_existing_sample_idx:
                    if abs(idx - k) < best_dis:
                        best_idx = k
                        best_dis = abs(idx - k)

                actual_sample_idx.append(best_idx)
                inrange_existing_sample_idx.remove(best_idx)

                if len(actual_sample_idx) == 1 / self._ratio:
                    break

        return sorted(actual_sample_idx)
