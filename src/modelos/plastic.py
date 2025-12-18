from abc import ABC, abstractmethod
from .tensortypes import Vetor6x1
from .elastic import Elastic
from .hardening import Hardening
import numpy as np
from numpy import float64
from numpy.typing import NDArray


class Plastic(Elastic, Hardening, ABC):

    @property
    def phi(self) -> float:
        """Angulo de atrito no espaco dos circulos de Mohr (`sigma`-`tau`)

        .. figure:: images/phi_M.png
        """
        return np.asin(3 * self.Mc / (6 + self.Mc))

    @property
    def Mc(self) -> float:
        """Inclinacao da envoltoria de cisalhamento `Mc` no espaco `p`-`q` (Linha
        de Estado Crítico)

        .. figure:: images/Mc_phi.png
        """
        return 6 * np.sin(self.phi) / (3 - np.sin(self.phi))

    @abstractmethod
    def func_plastica(self, *args) -> float:
        """Calcula e retorna o valor da funcao de plastificacao (`f`) em termos do
        estado de tensao.

        .. figure:: images/f_abs.png
        """
        raise NotImplementedError("Needs to implement")

    @abstractmethod
    def grad_f(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da funcao de plastificacao (`f`) em
        relacao ao estado de tensao.

        .. figure:: images/gradf_abs.png
        """
        raise NotImplementedError("Needs to implement")

    @abstractmethod
    def q_plastic(
        self, p: float | NDArray[float64], s: float | NDArray[float64]
    ) -> float | NDArray[float64]:
        """Calcula a tensao desviadora de plastificacao em funcao da tensao octaedrica.

        .. figure:: images/q_abs.png
        """
        raise NotImplementedError("Needs to implement")

    def grad_g(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da funcao potencial plastico (`g`) em
        relacao ao estado de tensao.

        Por default, igual ao gradiente da funcao de plastificacao (`f`):
        `grad_g = grad_f` (fluxo associado).

        .. figure:: images/gradg_abs.png
        """
        return self.grad_f()

    def dfds(self) -> float:
        """Calcula e retorna a derivada da funcao de plastificacao (`f`)
        em relacao a variavel interna de endurecimento do tipo tensao (s)

        Por default, igual a zero: `df/ds = 0` (sem endurecimento).

        .. figure:: images/dfds_abs.png
        """
        return 0.0

    def multiplicador_plastico(self, deps: Vetor6x1) -> float:
        """Calcula e retorna o multiplicador plastico.

        .. figure:: images/chi_abs.png
        """
        Df = self.grad_f()
        Dg = self.grad_g()
        De = self.matriz_elastica()
        return float(
            (
                (Df.T @ De @ deps)
                / (Df.T @ De @ Dg - self.dfds() * self.dsdh() * self.grad_h().T @ Dg)
            )[0, 0]
        )
