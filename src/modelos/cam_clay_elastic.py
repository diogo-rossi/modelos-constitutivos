from .elastic import Elastic


class CamClayElastico(Elastic):
    p: float
    e: float

    def __init__(self, k: float, v: float) -> None:
        self.poisson = v
        self.k = k

    ####################################################################################
    # %:          Metodos re-implementados
    ####################################################################################

    @property
    def K(self) -> float:
        """Modulo de elasticidade bulk volumetrica (`K`)

        Especificado para o Cam-Clay.

        .. figure:: images/Kb_cc.png
        """
        return (1 + self.e) * self.p / self.k

    @property
    def young(self) -> float:
        """Modulo de Young (`E`) do material.

        Especificado para o Cam-Clay. Em funcao do `K` e do Poisson.

        .. figure:: images/young_cc.png
        """
        return self.K * 3 * (1 - 2 * self.poisson)
