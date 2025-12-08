from .elastic import Elastic


class LinearElastico(Elastic):
    def __init__(self, E: float, v: float) -> None:
        self.young = E
        self.poisson = v
