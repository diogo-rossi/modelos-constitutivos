import numpy as np
from .elastic import Elastic
from .plastic import Plastic
from .hardening import Hardening
from .material import Material
from .tensortypes import Vetor6x1
from .funcs import epsV, vc
from numpy import float64
from numpy.typing import NDArray

##############################################################################################################
# %%          Elastico
##############################################################################################################


class CamClayElastico(Elastic):
    p: float
    e: float

    def __init__(self, k: float, v: float) -> None:
        self.poisson = v
        self.k = k

    ##########################################################################################################
    # %:          Metodos re-implementados
    ##########################################################################################################

    @property
    def young(self) -> float:
        """Modulo de Young (`E`) do material.

        Especificado para o Cam-Clay. Em funcao do `K` e do Poisson.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/young_cc.png
        """
        return self.K * 3 * (1 - 2 * self.poisson)

    @property
    def K(self) -> float:
        """Modulo de elasticidade bulk volumetrica (`K`)

        Especificado para o Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/Kb_cc.png
        """
        return (1 + self.e) * self.p / self.k


##############################################################################################################
# %%          Endurecimento
##############################################################################################################


class CamClayHardening(Hardening):

    def __init__(self, k: float, L: float, e0: float, p0: float) -> None:
        self.k = k
        self.L = L
        self.e = e0
        self.p0 = p0

    ##########################################################################################################
    # %:          Metodos re-implementados
    ##########################################################################################################

    @property
    def s(self) -> float:
        """Variavel interna de endurecimento do tipo tensao (`s`)

        Especificada para o Cam-Clay: Igual a tensao de pre-adensamento (`p0`).

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/s_cc.png
        """
        return self.p0

    @s.setter
    def s(self, value: float) -> None:
        self.p0 = value

    def dh(self, epsilonP: Vetor6x1) -> float:
        """Variacao da variavel interna de endurecimento do tipo
        deformacao (`h`) em funcao do estado de deformacao plastica.

        Especificada para o Cam-Clay: Igual a variacao da deformacao volumétrica plastica.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dh_cc.png
        """
        return self.depsVP(epsilonP)

    def grad_h(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da variavel interna de endurecimento
        do tipo deformacao (`h`) em funcao do estado de deformacao plastica.

        Especificada para o Cam-Clay. Igual ao gradiente da deformacao volumétrica plastica.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/gradh_cc.png
        """
        return self.grad_epsVP()

    def dsdh(self) -> float:
        """Calcula e retorna o modulo de endurecimento (`H`).

        Definido como a derivada da variavel interna de endurecimento do tipo tensao (`s`) em
        relacao a variavel interna de endurecimento do tipo deformacao (`h`).

        Especificado para o Cam-Clay: Igual a derivada da tensao de pre-adensamento (`s = p0`) em relacao a
        deformacao volumetrica plastica (`h = epsVP`).

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dsdh_cc.png
        """
        return self.dp0depsV()

    ##########################################################################################################
    # %:          Novos metodos
    ##########################################################################################################

    def depsVP(self, depsilonP: Vetor6x1) -> float:
        """Variacao da deformacao volumetrica plastica.

        Igual a variacao da variavel interna de endurecimento do tipo deformacao (`h`) do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/depsVP.png
        """
        return epsV(depsilonP)

    def grad_epsVP(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da deformacao volumétrica plastica.

        Igual ao vetor gradiente da variavel interna de endurecimento do tipo deformacao (`h`) do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/gradepsV_cc.png
        """
        return vc([1, 1, 1, 0, 0, 0])

    def dp0depsV(self) -> float:
        """Derivada da tensao de pre-adensamento (`p0`) em relacao a deformacao volumetrica plastica (`depsVP`)

        Igual ao modulo de endurecimento (`H`) especificado para o Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dp0depsV_cc.png
        """
        return ((1 + self.e) / (self.L - self.k)) * self.p0


##############################################################################################################
# %%          Plastificacao
##############################################################################################################


class CamClayPlastico(Plastic):
    s: float
    p: float
    q: float
    sigma: Vetor6x1

    def __init__(self, Mc: float, p0: float) -> None:
        self.Mc, self.p0 = Mc, p0

    @property
    def Mc(self) -> float:
        """Inclinacao da envoltoria de cisalhamento `Mc` no espaco `p`-`q` (Linha de Estado Crítico)

        Especificada para o Cam-Clay (dado de entrada).
        """
        return self._Mc

    @Mc.setter
    def Mc(self, value: float) -> None:
        self._Mc = value

    ##########################################################################################################
    # %:          Metodos re-implementados
    ##########################################################################################################

    def func_plastica(self) -> float:
        """Calcula e retorna o valor da funcao de plastificacao (`f`) em termos do estado de tensao.

        Especificada para o Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/f_cc.png
        """
        Mc, p, q, p0 = self.Mc, self.p, self.q, self.s
        return (Mc**2) * (p**2) - (Mc**2) * p0 * p + q**2

    def grad_f(self) -> Vetor6x1:
        """Calcula e retorna o vetor gradiente da funcao de plastificacao (`f`) em relacao ao estado de tensao.

        Especificada para o Cam-Clay, calculada por partes.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/gradf_cc.png
        """
        return self.dfdp() * self.grad_p() + self.dfdq() * self.grad_q()

    def q_plastic(self, p: float | NDArray[float64], s: float | NDArray[float64]) -> float | NDArray[float64]:
        """Calcula a tensao desviadora de plastificacao em funcao da tensao octaedrica.

        Especificada para o Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/q_cc.png
        """
        Mc, p0 = self.Mc, s
        return np.sqrt(np.abs((Mc**2) * p0 * p - (Mc**2) * (p**2)))

    def dfds(self):
        """Calcula e retorna a derivada da funcao de plastificacao (`f`)
        em relacao a variavel interna de endurecimento do tipo tensao (`s`)

        Especificada para o Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dfds_cc.png
        """
        return self.dfdp0()

    ##########################################################################################################
    # %:          Novos metodos
    ##########################################################################################################

    def dfdp(self) -> float:
        """Derivada de `f` em relacao a `p`, usada no calculo do `grad(f)` do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dfdp_cc.png
        """
        Mc, p0, p = self.Mc, self.p0, self.p
        return (Mc**2) * (2 * p - p0)

    def dfdq(self) -> float:
        """Derivada de `f` em relacao a `q`, usada no calculo do `grad(f)` do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dfdq_cc.png
        """
        q = self.q
        return 2 * q

    def grad_p(self) -> Vetor6x1:
        """Gradiente de `p` em relacao ao estado de tensao, usada no calculo do `grad(f)` do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/gradp_cc.png
        """
        return vc([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])

    def grad_q(self) -> Vetor6x1:
        """Gradiente de `q` em relacao ao estado de tensao, usada no calculo do `grad(f)` do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/gradq_cc.png
        """
        sigma, p, q = self.sigma, self.p, self.q
        if q < 0.00000001:
            return np.zeros((6, 1))
        return (3 / (2 * q)) * (sigma * vc([1, 1, 1, 2, 2, 2]) - p * vc([1, 1, 1, 0, 0, 0]))

    def dfdp0(self) -> float:
        """Derivada de `f` em relacao a `p0`, igual a `df/ds` do Cam-Clay.

        .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/dfdp0_cc.png
        """
        Mc, p = self.Mc, self.p
        return -(Mc**2) * p


##############################################################################################################
# %%          Classe principal
##############################################################################################################


class CamClay(Material, CamClayPlastico, CamClayElastico, CamClayHardening):
    """Classe de implementacao do Modelo Cam-Clay

    Parameters
    ----------
    - `k` (`float`):
        Parametro de elasticidade `kappa`.

    - `L` (`float`):
        Parametro de plasticidade `lambda`.

    - `v` (`float`):
        Razao de Possion `nu`.

    - `Mc` (`float`):
        Inclinacao `Mcrit` da envoltoria de ruptura no espaco `p-q`.

    - `e0` (`float`):
        Indice de vazios inicial `e0`.

    - `p0` (`float`):
        Tensao de pre-adensamento inicial `p0`.

    - `sigma0` (`Vetor6x1`):
        Estao de tensao inicial.

    - `epsilon0` (`Vetor6x1`):
        Estado de deformacao inicial.

    """

    def __init__(
        self,
        k: float,
        L: float,
        v: float,
        Mc: float,
        e0: float,
        p0: float,
        sigma0: Vetor6x1,
        epsilon0: Vetor6x1,
    ) -> None:

        super(CamClay, self).__init__(sigma0, epsilon0)
        super(Material, self).__init__(Mc=Mc, p0=p0)  # Plastico
        super(CamClayPlastico, self).__init__(k=k, v=v)  # Elastico
        super(CamClayElastico, self).__init__(k=k, L=L, e0=e0, p0=p0)  # Hardening
        self.update_state(sigma=sigma0)
