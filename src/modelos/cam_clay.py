from .material import Material
from .tensortypes import Vetor6x1
from .cam_clay_elastic import CamClayElastico
from .cam_clay_hardening import CamClayHardening
from .cam_clay_plastic import CamClayPlastico


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
