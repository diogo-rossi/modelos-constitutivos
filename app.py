import streamlitrunner as sr

sr.run()

##############################################################################################################
# %%          IMPORTS
##############################################################################################################

import streamlit as st
import numpy as np
from pandas import DataFrame
from typing import TypedDict, Any
from modelos.material import Material
from modelos.cam_clay import CamClay
from modelos.integracao import integra_deformacao
from modelos.funcs import vc
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter, Figure
from streamlit.delta_generator import DeltaGenerator  # for typing
from dataclasses import dataclass, asdict

##############################################################################################################
# %%          CONFIGURACOES
##############################################################################################################


class NomeValor(TypedDict):
    nome: str
    valor: float


class ParametrosClasse(TypedDict):
    parametros: dict[str, NomeValor]
    classe: type[Material]


slider_dict: dict[str, Any] = dict(active=0, currentvalue={"visible": False}, ticklen=0, pad={"t": 50})
axes_dict: dict[str, Any] = dict(mirror="allticks", ticks="inside", showgrid=True, title_standoff=5)
fig_dict: dict[str, Any] = dict(
    height=900,
    template="simple_white",
    plot_bgcolor="white",
    showlegend=True,
    margin=dict(l=50, r=0, t=30, b=0),
)

RUN_MODEL: str = "run_model"
DF: str = "DataFrame"

if RUN_MODEL not in st.session_state:
    st.session_state[RUN_MODEL] = True


def on_field_change():
    st.session_state[RUN_MODEL] = True


st.markdown(
    """
<style>
    .block-container {
        padding-top: 0rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

import streamlit as st

st.markdown(
    """
<style>
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Remove top padding from main container */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.set_page_config(layout="wide")

##############################################################################################################
# %%          MODELOS
##############################################################################################################

modelos: dict[str, ParametrosClasse] = {
    "Cam Clay": {
        "parametros": {
            "k": {"nome": "Kappa ($\\kappa$)", "valor": 0.01},
            "L": {"nome": "Lambda ($\\lambda$)", "valor": 0.1},
            "v": {"nome": "Poisson ($\\nu$)", "valor": 0.4},
            "Mc": {"nome": "$M_{crit}$", "valor": 1.1},
            "e0": {"nome": "Void ratio (e)", "valor": 0.9},
            "p0": {"nome": "$P_0$ (kPa)", "valor": 100},
        },
        "classe": CamClay,
    },
    "Mohr Coulomb": {
        "parametros": {
            "E": {"nome": "Young (E - kPa)", "valor": 50000},
            "v": {"nome": "Poisson ($\\nu$)", "valor": 0.2},
            "c": {"nome": "Cohesion (c)", "valor": 50},
            "Phi": {"nome": "$\\varphi$ (°)", "valor": 32},
            "Psi": {"nome": "$\\psi$ (°)", "valor": 4},
        },
        "classe": CamClay,
    },
}


##############################################################################################################
# %%          FUNCAO: ADICIONA PLOTS
##############################################################################################################


def add_plots(column: DeltaGenerator, df: DataFrame, material: Material, nP: int = 1000, nS: int = 1000):
    N: int = len(df)
    n = column.slider("Step", min_value=0, max_value=N - 1, value=0, step=1)

    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.05, vertical_spacing=0.07)

    x_labels = np.array(
        [
            ["ln(p)", "p (kPa)"],
            ["Deformacao", "Tensao normal (kPa)"],
        ]
    )
    y_labels = np.array(
        [
            ["Indice de vazios (e)", "q (kPa)"],
            ["Tensao (kPa)", "Tensao cisalhante (kPa)"],
        ]
    )

    for i in range(2):
        for j in range(2):
            fig.update_xaxes(row=i + 1, col=j + 1, title_text=x_labels[i, j], **axes_dict)
            fig.update_yaxes(row=i + 1, col=j + 1, title_text=y_labels[i, j], **axes_dict)

    ##########################################################################################################
    # %           INDICE DE VAZIOS POR P
    ##########################################################################################################

    fig.update_xaxes(row=1, col=1, type="log", range=[np.log10(np.min(df.p)), np.log10(np.max(df.p))])
    fig.add_trace(row=1, col=1, trace=Scatter(x=df.p, y=df.e, showlegend=False, name="e vs lnP"))
    add_marker(fig, 1, 1, df.p[n], df.e[n])

    ##########################################################################################################
    # %           P x Q
    ##########################################################################################################

    pEnv = np.linspace(0, np.max(df.s), 100)
    qEnv = material.Mc * pEnv
    fig.add_trace(row=1, col=2, trace=Scatter(x=df.p, y=df.q, showlegend=False, name="Trajetoria"))
    fig.add_trace(row=1, col=2, trace=Scatter(x=pEnv, y=qEnv, showlegend=False, name="Envoltoria"))
    fig.update_xaxes(row=1, col=2, range=[0, np.max(df.s)])
    fig.update_yaxes(row=1, col=2, range=[0, np.max(qEnv)])

    p0 = np.array(df.s).reshape(N, 1)
    p = p0 * np.linspace(0, 1, nP)
    q = np.array(material.q_plastic(p, p0))

    fig.add_trace(row=1, col=2, trace=Scatter(x=p[0], y=q[0], showlegend=False, name="Sup-Plast-Ini"))
    fig.add_trace(row=1, col=2, trace=Scatter(x=p[n], y=q[n], showlegend=False, name="Sup. Plastif."))

    add_marker(fig, 1, 2, df.p[n], df.q[n])

    ##########################################################################################################
    # %           TENSAO x DEFORMACAO
    ##########################################################################################################
    S1max = np.max(df.S1)
    fig.update_yaxes(row=2, col=1, range=[0, S1max])
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ex, y=df.Sx, name="Sx-Ex"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ex, y=df.Sy, name="Sy-Ex"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ex, y=df.Sz, name="Sz-Ex"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ey, y=df.Sx, name="Sx-Ey"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ey, y=df.Sy, name="Sy-Ey"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ey, y=df.Sz, name="Sz-Ey"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ez, y=df.Sx, name="Sx-Ez"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ez, y=df.Sy, name="Sy-Ez"))
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ez, y=df.Sz, name="Sz-Ez"))

    add_marker(fig, 2, 1, df.Ex[n], df.Sx[n])
    add_marker(fig, 2, 1, df.Ex[n], df.Sy[n])
    add_marker(fig, 2, 1, df.Ex[n], df.Sz[n])
    add_marker(fig, 2, 1, df.Ey[n], df.Sx[n])
    add_marker(fig, 2, 1, df.Ey[n], df.Sy[n])
    add_marker(fig, 2, 1, df.Ey[n], df.Sz[n])
    add_marker(fig, 2, 1, df.Ez[n], df.Sx[n])
    add_marker(fig, 2, 1, df.Ez[n], df.Sy[n])
    add_marker(fig, 2, 1, df.Ez[n], df.Sz[n])

    ##########################################################################################################
    # %           SIGMA x TAU
    ##########################################################################################################

    sig = np.linspace(0, S1max, 100)
    tau = np.tan(material.phi) * sig
    fig.add_trace(row=2, col=2, trace=Scatter(x=sig, y=tau, showlegend=False, name="Envoltoria"))
    fig.update_yaxes(row=2, col=2, range=[0, S1max])

    S1 = np.array(df.S1).reshape(N, 1)
    S3 = np.array(df.S3).reshape(N, 1)
    R = (S1 - S3) / 2
    C = (S1 + S3) / 2
    S = S3 + (S1 - S3) * np.linspace(0, 1, nS)
    T = np.sqrt(np.abs(R**2 - (S - C) ** 2))
    fig.add_trace(row=2, col=2, trace=Scatter(x=S[n], y=T[n], showlegend=False, name="Circ. de Mohr"))

    ##########################################################################################################
    # %           ADICIONA FIGURA
    ##########################################################################################################

    fig.update_layout(**fig_dict)

    column.plotly_chart(fig, width="stretch", theme=None)


def add_marker(fig: Figure, row: int, col: int, x: float, y: float):
    """Adiciona marcador do ensaio na curva"""
    fig.add_trace(
        row=row,
        col=col,
        trace=Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name="Estado atual",
            marker=dict(size=8, color="red"),
            showlegend=False,
        ),
    )


##############################################################################################################
# %%          FUNCAO: CRIA A INTERFACE
##############################################################################################################


def setup_app(modelos: dict[str, ParametrosClasse]):

    ##########################################################################################################
    # %           COLUNAS, BOTAO DE FECHAR E MODELOS
    ##########################################################################################################

    left, right = st.columns([24, 100])
    fechar, modelo = left.columns(2)
    fechar.button("Close", on_click=sr.close_app)
    modelo = modelo.selectbox("Modelo", [name for name in modelos], on_change=on_field_change)

    ##########################################################################################################
    # %           PARAMETROS
    ##########################################################################################################

    parametros = modelos[modelo]["parametros"]

    k = 2
    for par in parametros:
        k += 1
        if k >= 2:
            k = 0
            cols = left.columns(2)
        parametros[par]["valor"] = cols[k].number_input(
            label=parametros[par]["nome"], value=parametros[par]["valor"], on_change=on_field_change
        )

    ##########################################################################################################
    # %           ESTADO INICIAL
    ##########################################################################################################

    left.text("Estado inicial")
    k = 3
    stress: list[float] = [10, 10, 10, 0, 0, 0]
    var = ("sigma", "tau")
    for i, s in enumerate(["x", "y", "z", "xy", "yz", "zx"]):
        k += 1
        if k >= 3:
            k = 0
            cols = left.columns(3)
        stress[i] = cols[k].number_input(
            f"$\\{var[0] if i<3 else var[1]}_{{{s}0}}$", stress[i], on_change=on_field_change
        )

    ##########################################################################################################
    # %           TIPO DE ENSAIOS E NUMERO DE STEPS
    ##########################################################################################################

    ensaio, steps = left.columns(2)
    ensaio = ensaio.selectbox("Ensaio", ["Oedometrico", "Undrained"], on_change=on_field_change)
    steps = steps.number_input("Steps", value=1000, on_change=on_field_change)

    ##########################################################################################################
    # %           INCREMENTOS
    ##########################################################################################################

    incrementos: list[float] = [0.0, 0.0001, 0.0, 0.0, 0.0, 0.0]
    controle = left.selectbox("Controle", ["Deformacao", "Tensao"], on_change=on_field_change)
    var = ("sigma", "tau") if controle == "Tensao" else ("varepsilon", "gamma")
    k = 3
    for i, s in enumerate(["x", "y", "z", "xy", "yz", "zx"]):
        k += 1
        if k >= 3:
            k = 0
            cols = left.columns(3)
        incrementos[i] = cols[k].number_input(
            f"d$\\{var[0] if i<3 else var[1]}_{{{s}}}$",
            value=incrementos[i],
            format=("%.5f" if controle == "Deformacao" else "%.2f"),
            on_change=on_field_change,
        )

    ##########################################################################################################
    # %           DEFINICAO DO MODELO
    ##########################################################################################################

    sig0 = vc(stress)
    eps = vc([0, 0, 0, 0, 0, 0])
    kwargs: dict[str, Any] = {k: parametros[k]["valor"] for k in parametros}
    kwargs.update(sigma0=sig0, epsilon0=eps)
    material = modelos[modelo]["classe"](**kwargs)

    if st.session_state.get(RUN_MODEL, False):
        st.session_state[DF] = integra_deformacao(deps=vc(incrementos), num_steps=steps, material=material)
        st.session_state[DF].to_csv("resultadoApp.csv")
        st.session_state[RUN_MODEL] = False

    add_plots(right, st.session_state[DF], material)


##############################################################################################################
# %%          EXECUCAO PRINCIPAL
##############################################################################################################

if __name__ == "__main__":
    setup_app(modelos)
