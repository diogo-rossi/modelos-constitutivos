from .plastic import Plastic
from .tensortypes import Vetor6x1
from numpy import float64
from numpy.typing import NDArray
from .funcs import vc
import numpy as np


class CamClayPlastico(Plastic):
    s: float
    p: float
    q: float
    sigma: Vetor6x1

    def __init__(self, Mc: float, p0: float) -> None:
        self.Mc, self.p0 = Mc, p0

    @property
    def Mc(self) -> float:
        """Inclinacao da envoltoria de cisalhamento `Mc` no espaco `p`-`q` (Linha de
        Estado Crítico)

        Especificada para o Cam-Clay (dado de entrada).
        """
        return self._Mc

    @Mc.setter
    def Mc(self, value: float) -> None:
        self._Mc = value

    ####################################################################################
    # %:          Metodos re-implementados
    ####################################################################################

    def func_plastica(self) -> float:
        """Calcula e retorna o valor da funcao de plastificacao (`f`) em termos do
        estado de tensao.

        Especificada para o Cam-Clay.

        .. figure:: images/f_cc.png
        """
        Mc, p, q, p0 = self.Mc, self.p, self.q, self.s
        return (Mc**2) * (p**2) - (Mc**2) * p0 * p + q**2

    def grad_f(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da funcao de plastificacao (`f`) em
        relacao ao estado de tensao.

        Especificada para o Cam-Clay, calculada por partes.

        .. figure:: images/gradf_cc.png
        """
        return self.dfdp() * self.grad_p() + self.dfdq() * self.grad_q()

    def q_plastic(
        self, p: float | NDArray[float64], s: float | NDArray[float64]
    ) -> float | NDArray[float64]:
        """Calcula a tensao desviadora de plastificacao em funcao da tensao octaedrica.

        Especificada para o Cam-Clay.

        .. figure:: images/q_cc.png
        """
        Mc, p0 = self.Mc, s
        return np.sqrt(np.abs((Mc**2) * p0 * p - (Mc**2) * (p**2)))

    def dfds(self):
        """Calcula e retorna a derivada da funcao de plastificacao (`f`)
        em relacao a variavel interna de endurecimento do tipo tensao (`s`)

        Especificada para o Cam-Clay.

        .. figure:: images/dfds_cc.png
        """
        return self.dfdp0()

    ####################################################################################
    # %:          Novos metodos
    ####################################################################################

    def dfdp(self) -> float:
        """Derivada de `f` em relacao a `p`, usada no calculo do `grad(f)` do Cam-Clay.

        .. figure:: images/dfdp_cc.png
        """
        Mc, p0, p = self.Mc, self.p0, self.p
        return (Mc**2) * (2 * p - p0)

    def dfdq(self) -> float:
        """Derivada de `f` em relacao a `q`, usada no calculo do `grad(f)` do Cam-Clay.

        .. figure:: images/dfdq_cc.png
        """
        q = self.q
        return 2 * q

    def grad_p(self) -> Vetor6x1:
        """Gradiente de `p` em relacao ao estado de tensao, usada no calculo do
        `grad(f)` do Cam-Clay.

        .. figure:: images/gradp_cc.png
        """
        return vc([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])

    def grad_q(self) -> Vetor6x1:
        """Gradiente de `q` em relacao ao estado de tensao, usada no calculo do
        `grad(f)` do Cam-Clay.

        .. figure:: images/gradq_cc.png
        """
        sigma, p, q = self.sigma, self.p, self.q
        if q < 0.00000001:
            return np.zeros((6, 1))
        return (3 / (2 * q)) * (
            sigma * vc([1, 1, 1, 2, 2, 2]) - p * vc([1, 1, 1, 0, 0, 0])
        )

    def dfdp0(self) -> float:
        """Derivada de `f` em relacao a `p0`, igual a `df/ds` do Cam-Clay.

        .. figure:: images/dfdp0_cc.png
        """
        Mc, p = self.Mc, self.p
        return -(Mc**2) * p
