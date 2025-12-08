from numpy.typing import NDArray
from numpy import float64

Vetor6 = NDArray[float64]
Vetor6x1 = NDArray[float64]
Vetor1x6 = NDArray[float64]
Matriz3x3 = NDArray[float64]
Tensor = Vetor6 | Vetor1x6 | Vetor6x1 | Matriz3x3
Matriz6x6 = NDArray[float64]
