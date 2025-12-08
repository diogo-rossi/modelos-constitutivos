import numpy as np
from .tensortypes import Vetor1x6, Vetor6x1, Vetor6, Matriz3x3


def vetor(vec_or_tensor: Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3) -> Vetor6:
    """Retorna um vetor de 6 elementos de ordem 1. Aceita varias entradas"""
    if vec_or_tensor.shape in [(6, 1), (1, 6), (6,)]:
        return vec_or_tensor.reshape((6,))
    if vec_or_tensor.shape == (3, 3):
        return np.concat(
            [vec_or_tensor.diagonal(0), vec_or_tensor.diagonal(1), vec_or_tensor.diagonal(3)]
        ).reshape((6,))
    raise ValueError("Vetor/Tensor errado")


def tensor(vec_or_tensor: Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3) -> Matriz3x3:
    """Retorna um tensor simetrico 3x3. Aceita varias entradas"""
    if vec_or_tensor.shape in [(6, 1), (1, 6), (6,)]:
        vec_or_tensor = vec_or_tensor.reshape((6,))
        return np.array(
            [
                [vec_or_tensor[0], vec_or_tensor[3], vec_or_tensor[5]],
                [vec_or_tensor[3], vec_or_tensor[1], vec_or_tensor[4]],
                [vec_or_tensor[5], vec_or_tensor[4], vec_or_tensor[2]],
            ]
        )
    if vec_or_tensor.shape == (3, 3):
        return vec_or_tensor
    raise ValueError("Vetor/Tensor errado")


def vetor_para_linha(vec: Vetor6x1) -> list[float]:
    """Retorna uma lista de numeros, usando um vetor"""
    return [float(s) for s in vetor(vec)]


def vc(vec: list[float | int]) -> Vetor6x1:
    """Retorna um vetor coluna, usando uma lista de numeros"""
    return np.array([[float(v) for v in vec]]).T


def trace(vec_or_tensor: Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3) -> float:
    """Retorna o 'traco' de um vetor 6x6 (3 primeiros elementos) ou soma da diagonal da matriz 3x3"""
    return float(vetor(vec_or_tensor)[0:3].sum())


def octa(sigma: Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3) -> float:
    """Retora a tensao octaedrica, igual a media: `(Sx + Sy + Sz) / 3`"""
    return trace(sigma) / 3


def epsV(epsilon: Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3) -> float:
    """Retora a deformacao volumetrica, igual a soma: `epsX + epsY + epsZ`"""
    return trace(epsilon)


def desv(sigma: Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3) -> float:
    """Retorna a tensao desviadora, dada pela formula abaixo:
    
    .. figure:: file:///C:/Users/rossi/0/repos/Modelos-constitutivos/src/modelos/images/desv.png
    """
    sx, sy, sz, txy, tyz, tzx = vetor(sigma)
    return float(
        np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6 * (txy**2 + tyz**2 + tzx**2)))
    )
