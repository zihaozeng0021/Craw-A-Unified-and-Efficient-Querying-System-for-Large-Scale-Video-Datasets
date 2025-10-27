class F1Evaluator:
    def __init__(self, pred):
        self._pred = pred

    def evaluate(self, res, ref_res):
        true_pos, pos = 0, 0

        for i in ref_res:
            if i not in res:
                raise Exception("{} not in res".format(i))

            if self._pred.evaluate(res[i]):
                pos += 1
                if self._pred.evaluate(ref_res[i]):
                    true_pos += 1

        precision = (true_pos / pos) if pos != 0 else 0

        pos = 0

        for i in ref_res:
            if self._pred.evaluate(ref_res[i]):
                pos += 1

        recall = (true_pos / pos) if pos != 0 else 0

        if precision == 0 or recall == 0:
            res = 0
        else:
            res = 2 / (1 / precision + 1 / recall)

        return res
