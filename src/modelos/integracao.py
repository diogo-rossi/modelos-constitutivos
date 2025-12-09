from pandas import DataFrame
from .tensortypes import Vetor6x1
from .material import Material
from .funcs import vc


def integra_deformacao(
    deps: Vetor6x1,
    num_steps: int,
    material: Material,
) -> DataFrame:
    """Integra a deformacao no material, calculando as tensoes.

    Retorna um `DataFrame` do `pandas` com cada linha representando um estágio de
    carregamento de deformação.

    Ordem dos valores de retorno:

    `Sx`, `Sy`, `Sz`, `Txy`, `Tyz`, `Tzx`, `Ex`, `Ey`, `Ez`, `Gxy`, `Gyz`, `Gzx`,
    `e`, `s`, `Ev`, `Eq`, `Eve`, `Eqe`, `Evp`, `Eqp`, `p`, `q`, `f`, `qui`,
    `S1`, `S2`, `S3`
    """

    tabela: list[list[float]] = [material.apply_strain(vc([0, 0, 0, 0, 0, 0]))]

    for i in range(num_steps):

        tabela.append(material.apply_strain(deps))

    colunas: list[str] = [
        "Sx",
        "Sy",
        "Sz",
        "Txy",
        "Tyz",
        "Tzx",
        "Ex",
        "Ey",
        "Ez",
        "Gxy",
        "Gyz",
        "Gzx",
        "e",
        "s",
        "Ev",
        "Eq",
        "Eve",
        "Eqe",
        "Evp",
        "Eqp",
        "p",
        "q",
        "f",
        "chi",
        "S1",
        "S2",
        "S3",
    ]

    return DataFrame(tabela, columns=colunas)
