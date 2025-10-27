class Chunk:
    def __init__(self):
        # specify the chunk range [start, end)
        self._start = -1
        self._end = -1

        self._model = None

    @property
    def range(self):
        return self._start, self._end

    @property
    def model(self):
        return self._model

    def set_range(self, start, end):
        self._start = start
        self._end = end

    def set_model(self, model):
        self._model = model

    def split(self):
        span = self._end - self._start
        half_span = span // 2

        l_chunk = Chunk()
        l_chunk.set_range(self._start, self._start + half_span)

        r_chunk = Chunk()
        r_chunk.set_range(self._start + half_span, self._end)

        return l_chunk, r_chunk