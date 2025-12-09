from .elastic import Elastic
from .hardening import Hardening
from .plastic import Plastic
from .tensortypes import Vetor6x1
from .funcs import octa, desv, epsV, vetor_para_linha, tensor
import numpy as np


class Material(Plastic, Elastic, Hardening):
    """Classe do modelo de Material, que pode possuir os metodos dos modelos:

    - `Elastic`
    - `Hardening`
    - `Plastic`
    """

    def __init__(self, sigma0: Vetor6x1, epsilon0: Vetor6x1) -> None:
        self.sigma = sigma0
        self.epsilon = epsilon0

    def update_state(
        self,
        sigma: Vetor6x1,
        depsilon: Vetor6x1 | None = None,
        depsilonP: Vetor6x1 | None = None,
    ) -> None:
        """Atualiza as variaveis de estado do material"""

        self.sigma = sigma
        self.p = octa(self.sigma)
        self.q = desv(self.sigma)
        if depsilon is not None:
            self.e -= (1 + self.e) * epsV(depsilon)
            self.epsilon += depsilon
        if depsilonP is not None:
            self.update_hardening(depsilonP)

    def apply_strain(self, deps: Vetor6x1) -> list[float]:
        """Aplica incremento de deformação e retorna novo estado
        do material em uma lista de valores

        Ordem dos valores de retorno:
        `Sx`, `Sy`, `Sz`, `Txy`, `Tyz`, `Tzx`, `Ex`, `Ey`, `Ez`, `Gxy`, `Gyz`, `Gzx`,
        `e`, `s`, `Ev`, `Eq`, `Eve`, `Eqe`, `Evp`, `Eqp`, `p`, `q`, `f`, `qui`,
        `S1`, `S2`, `S3`
        """
        sig = self.sigma

        # Tentativa elastica
        De = self.matriz_elastica()
        dsig = De @ deps
        sig += dsig

        # Atualiza estado de tensao e grava
        self.update_state(sig)
        # TODO: verificar se precisa fazer isso (isso atualiza a De).

        # Correcao plastica
        f: float = self.func_plastica()
        depsP: Vetor6x1 = np.zeros((6, 1))
        chi: float = 0
        if f > 0:
            chi: float = self.multiplicador_plastico(deps)
            nablaG: Vetor6x1 = self.grad_g()
            depsP: Vetor6x1 = chi * nablaG
            sig -= De @ depsP

        self.update_state(sig, deps, depsP)

        # Deformacoes volumetrica e desviadora
        depsV = epsV(deps)
        depsQ = desv(deps)
        depsVP = epsV(depsP)
        depsQP = desv(depsP)
        depsVE = depsV - depsVP
        depsQE = depsQ - depsQP

        S3, S2, S1 = sorted(
            [float(s) for s in np.linalg.eig(tensor(self.sigma)).eigenvalues]
        )

        return (
            vetor_para_linha(self.sigma)
            + vetor_para_linha(self.epsilon)
            + [
                self.e,
                self.s,
                depsV,
                depsQ,
                depsVE,
                depsQE,
                depsVP,
                depsQP,
                self.p,
                self.q,
                f,
                chi,
                S1,
                S2,
                S3,
            ]
        )
