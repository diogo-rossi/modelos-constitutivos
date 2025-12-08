import numpy as np
from abc import ABC
from .tensortypes import Matriz6x6


class Elastic(ABC):
    young: float
    poisson: float

    @property
    def K(self) -> float:
        """Modulo de elasticidade bulk volumetrica (`K`)

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/Kb.png
        """
        return self.young / (3 * (1 - 2 * self.poisson))

    @property
    def Me(self) -> float:
        """Modulo de elasticidade uniaxial `Me` (oedometrica).

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/Me.png
        """
        E, v = self.young, self.poisson
        return E * (1 - v) / ((1 + v) * (1 - 2 * v))

    @property
    def Ge(self) -> float:
        """Modulo de elasticidade cisalhante `Ge`.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/Ge.png
        """
        E, v = self.young, self.poisson
        return E / (2 * (1 + v))

    @property
    def Le(self) -> float:
        """Constante de Lamme da elasticidade (`Le`).

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/Le.png
        """
        M, G = self.Me, self.Ge
        return M - 2 * G

    def matriz_elastica(self) -> Matriz6x6:
        """Retorna a matriz de rigidez elastica `[De]`.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/De.png
        """
        M = self.Me
        G = self.Ge
        L = self.Le
        return np.array(
            [
                [M, L, L, 0, 0, 0],
                [L, M, L, 0, 0, 0],
                [L, L, M, 0, 0, 0],
                [0, 0, 0, G, 0, 0],
                [0, 0, 0, 0, G, 0],
                [0, 0, 0, 0, 0, G],
            ]
        )
