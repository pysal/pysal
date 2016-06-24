    def __new__(cls, y=None, w=None, **kwargs):
        if y is not None and w is not None:
            self = object.__new__(Geary)
            return self
        elif y is None and w is not None:
            def promise(y):
                return cls(y, w, **kwargs)
            return promise
        elif y is not None and w is None:
            def promise(w):
                return cls(y, w, **kwargs)
            return promise
        elif y is None and w is None:
            return cls
