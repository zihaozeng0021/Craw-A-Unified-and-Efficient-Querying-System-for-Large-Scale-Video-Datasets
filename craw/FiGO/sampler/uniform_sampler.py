class UniformSampler:
    def __init__(self, loader, ratio=0.1):
        self._loader = loader
        self._ratio = ratio

    def get_idx(self):
        length = len(self._loader)

        span = int(1 / self._ratio)
        return [i for i in range(0, length, span)]
