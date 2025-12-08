import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.axes import Axes
from modelos.funcs import vc
from modelos.integracao import integra_deformacao
from modelos.cam_clay import CamClay


if __name__ == "__main__":

    lamba = 0.1
    kappa = 0.01
    poisson = 0.4
    M = 1.1
    e0 = 0.9
    p0 = 100
    sig = vc([10, 10, 10, 0, 0, 0])
    eps = np.zeros((6, 1))
    dsig = np.zeros((6, 1))
    deps = vc([0, 0.0001, 0, 0, 0, 0])
    num_steps = 1000

    fig, ax = plt.subplots(1, 2)
    axvr: Axes = ax[0]
    axpq: Axes = ax[1]

    material = CamClay(k=kappa, L=lamba, v=poisson, Mc=M, e0=e0, p0=p0, sigma0=sig, epsilon0=eps)

    df: DataFrame = integra_deformacao(deps, num_steps, material)

    df.to_csv("resultado.csv")
    p = np.array(df["p"].values)
    q = np.array(df["q"].values)
    e = np.array(df["e"].values)
    lnP = np.log(p)
    sy = np.array(df["Sy"].values)

    axvr.plot(p, e)
    axvr.set_xscale("log")
    axvr.set_ylim(0.5, 1)

    p0 = np.array(df.s).reshape(len(df.s), 1)
    p0 = p0 * np.ones(100)
    pP = p0 * np.linspace(0, 1, 100)
    qP = np.array(material.q_plastic(pP, p0))

    axpq.plot(p, q, "x-")
    axpq.plot(pP[800], qP[800], "-")
    axpq.set_xlim(0, 500)
    axpq.set_ylim(0, M * 500)
    plt.show()
