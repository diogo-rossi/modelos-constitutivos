from .hardening import Hardening
from .tensortypes import Vetor6x1
from .funcs import vc, epsV


class CamClayHardening(Hardening):

    def __init__(self, k: float, L: float, e0: float, p0: float) -> None:
        self.k = k
        self.L = L
        self.e = e0
        self.p0 = p0

    ####################################################################################
    # %:          Metodos re-implementados
    ####################################################################################

    @property
    def s(self) -> float:
        """Variavel interna de endurecimento do tipo tensao (`s`)

        Especificada para o Cam-Clay: Igual a tensao de pre-adensamento (`p0`).

        .. figure:: images/s_cc.png
        """
        return self.p0

    @s.setter
    def s(self, value: float) -> None:
        self.p0 = value

    def dh(self, epsilonP: Vetor6x1) -> float:
        """Variacao da variavel interna de endurecimento do tipo
        deformacao (`h`) em funcao do estado de deformacao plastica.

        Especificada para o Cam-Clay: Igual variacao da deformacao volumétrica plastica.

        .. figure:: images/dh_cc.png
        """
        return self.depsVP(epsilonP)

    def grad_h(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da variavel interna de endurecimento
        do tipo deformacao (`h`) em funcao do estado de deformacao plastica.

        Especificada para o Cam-Clay. Igual ao grad. da deformacao volumétrica plastica.

        .. figure:: images/gradh_cc.png
        """
        return self.grad_epsVP()

    def dsdh(self) -> float:
        """Calcula e retorna o modulo de endurecimento (`H`).

        Definido como a derivada da variavel interna de endurecimento do tipo tensao
        (`s`) em relacao a variavel interna de endurecimento do tipo deformacao (`h`).

        Especificado para o Cam-Clay: Igual a derivada da tensao de pre-adensamento (`s
        = p0`) em relacao a deformacao volumetrica plastica (`h = epsVP`).

        .. figure:: images/dsdh_cc.png
        """
        return self.dp0depsV()

    ####################################################################################
    # %:          Novos metodos
    ####################################################################################

    def depsVP(self, depsilonP: Vetor6x1) -> float:
        """Variacao da deformacao volumetrica plastica.

        Igual a variacao da variavel interna de endurecimento do tipo deformacao (`h`)
        do Cam-Clay.

        .. figure:: images/depsVP.png
        """
        return epsV(depsilonP)

    def grad_epsVP(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da deformacao volumétrica plastica.

        Igual ao vetor gradiente da variavel interna de endurecimento do tipo deformacao
        (`h`) do Cam-Clay.

        .. figure:: images/gradepsV_cc.png
        """
        return vc([1, 1, 1, 0, 0, 0])

    def dp0depsV(self) -> float:
        """Derivada da tensao de pre-adensamento (`p0`) em relacao a deformacao
        volumetrica plastica (`depsVP`)

        Igual ao modulo de endurecimento (`H`) especificado para o Cam-Clay.

        .. figure:: images/dp0depsV_cc.png
        """
        return ((1 + self.e) / (self.L - self.k)) * self.p0
