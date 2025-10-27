class CountPredicate:
    def __init__(self, cls, count, score=0.5):
        # list of class and coresponding count
        self._cls = cls
        self._count = count

        # confidence score to filter detected objects
        self._score = score

    @property
    def cls(self):
        return self._cls

    @property
    def count(self):
        return self._count

    def evaluate(self, res):
        for i in range(len(self._cls)):
            per_cls, per_count = self._cls[i], self._count[i]

            cls_count = 0
            for k, c in enumerate(res['class']):
                if c == per_cls and res['score'][k] >= self._score:
                    cls_count += 1

            if cls_count < per_count:
                return False

        return True

    def __str__(self):
        return "Class: {}, Count: {}".format(self._cls, self._count)
            