from abc import ABC, abstractmethod
from .funcs import vc

from .tensortypes import Vetor6x1


class Hardening(ABC):
    s: float = 0.0
    """Variavel interna de endurecimento do tipo tensao (`s`)"""

    def dh(self, epsilonP: Vetor6x1) -> float:
        """Variacao da variavel interna de endurecimento do tipo
        deformacao (`h`) em funcao do estado de deformacao plastica.

        Por default, igual a zero: `dh = 0`.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dh_abs.png
        """
        return 0.0

    def grad_h(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da variavel interna de endurecimento
        do tipo deformacao (`h`) em funcao do estado de deformacao plastica.

        Por default, igual a um vetor nulo: `grad_h = {0,0,0,0,0,0}`.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/gradh_abs.png
        """
        return vc([0, 0, 0, 0, 0, 0])

    def dsdh(self) -> float:
        """Calcula e retorna o modulo de endurecimento (`H`).

        Definido como a derivada da variavel interna de endurecimento do tipo tensao (`s`) em
        relacao a variavel interna de endurecimento do tipo deformacao (`h`).

        Por default, igual a zero: `ds/dh = 0`.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dsdh_abs.png
        """
        return 0.0

    def update_hardening(self, epsilonP: Vetor6x1) -> None:
        """Atualiza a variavel interna de endurecimento do tipo tensao (`s`) usando a sua derivada

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/s_abs.png
        """
        self.s += self.dsdh() * self.dh(epsilonP)
