import streamlitrunner as sr

sr.run(title="Modelos constitutivos", maximized=True)

########################################################################################
# %%          IMPORTS
########################################################################################

import streamlit as st
import numpy as np
from pandas import DataFrame
from typing import TypedDict, Any
from modelos.material import Material
from modelos.cam_clay import CamClay
from modelos.integracao import integra_deformacao
from modelos.funcs import vc, desv, octa
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter, Figure, Mesh3d, Scatter3d, Frame
from streamlit.delta_generator import DeltaGenerator  # for typing
from skimage.measure import marching_cubes
from tqdm import tqdm

########################################################################################
# %%          CONFIGURACOES
########################################################################################


class NomeValor(TypedDict):
    nome: str
    valor: float


class ParametrosClasse(TypedDict):
    parametros: dict[str, NomeValor]
    classe: type[Material]


slider_dict: dict[str, Any] = dict(
    active=0, currentvalue={"visible": False}, ticklen=0, pad={"t": 50}
)
axes_dict: dict[str, Any] = dict(
    mirror="allticks", ticks="inside", showgrid=True, title_standoff=5
)
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

########################################################################################
# %%          MODELOS
########################################################################################

modelos: dict[str, ParametrosClasse] = {
    "Cam Clay": {
        "parametros": {
            "k": {"nome": "Kappa ($\\kappa$)", "valor": 0.01},
            "L": {"nome": "Lambda ($\\lambda$)", "valor": 0.1},
            "v": {"nome": "Poisson ($\\nu$)", "valor": 0.4},
            "Mc": {"nome": "$M_{crit}$", "valor": 1.1},
            "e0": {"nome": "Void ratio inicial ($e_0$)", "valor": 0.9},
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


def add_marker(
    fig: Figure,
    row: int,
    col: int,
    x: float,
    y: float,
    name: str | None = None,
    show=False,
):
    """Adiciona marcador do ensaio na curva"""
    name = name or "Estado atual"
    fig.add_trace(
        row=row,
        col=col,
        trace=Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name=name,
            marker=dict(size=8, color="red"),
            showlegend=show,
        ),
    )


########################################################################################
# %%          FUNCAO: CRIA A INTERFACE
########################################################################################


def setup_app(modelos: dict[str, ParametrosClasse]):

    ####################################################################################
    # %           COLUNAS, BOTAO DE FECHAR E MODELOS
    ####################################################################################

    left, right = st.columns([24, 100])
    modelo, steps = left.columns(2)
    modelo = modelo.selectbox(
        "Modelo", [name for name in modelos], on_change=on_field_change
    )

    steps = steps.number_input("Steps", value=1000, on_change=on_field_change)

    ####################################################################################
    # %           PARAMETROS
    ####################################################################################

    parametros = modelos[modelo]["parametros"]

    k = 2
    for par in parametros:
        k += 1
        if k >= 2:
            k = 0
            cols = left.columns(2)
        parametros[par]["valor"] = cols[k].number_input(
            label=parametros[par]["nome"],
            value=parametros[par]["valor"],
            on_change=on_field_change,
        )

    ####################################################################################
    # %           ESTADO INICIAL
    ####################################################################################

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
            f"$\\{var[0] if i<3 else var[1]}_{{{s}0}}$",
            stress[i],
            on_change=on_field_change,
        )

    ####################################################################################
    # %           INCREMENTOS
    ####################################################################################

    left.text("Incrementos")
    incrementos: list[float] = [0.0, 0.0001, 0.0, 0.0, 0.0, 0.0]
    var = ("sigma", "tau")
    k = 3
    for i, s in enumerate(["x", "y", "z", "xy", "yz", "zx"]):
        k += 1
        if k >= 3:
            k = 0
            cols = left.columns(3)
        incrementos[i] = cols[k].number_input(
            f"d$\\{var[0] if i<3 else var[1]}_{{{s}}}$",
            value=incrementos[i],
            format="%.2f",
            on_change=on_field_change,
        )

    var = ("varepsilon", "gamma")
    for i, s in enumerate(["x", "y", "z", "xy", "yz", "zx"]):
        k += 1
        if k >= 3:
            k = 0
            cols = left.columns(3)
        incrementos[i] = cols[k].number_input(
            f"d$\\{var[0] if i<3 else var[1]}_{{{s}}}$",
            value=incrementos[i],
            format="%.5f",
            on_change=on_field_change,
        )

    ####################################################################################
    # %           DEFINICAO DO MODELO
    ####################################################################################

    with left:
        with st.spinner("Running", show_time=True):

            sig0 = vc(stress)
            eps = vc([0, 0, 0, 0, 0, 0])
            kwargs: dict[str, Any] = {k: parametros[k]["valor"] for k in parametros}
            kwargs.update(sigma0=sig0, epsilon0=eps)
            material = modelos[modelo]["classe"](**kwargs)

            if st.session_state.get(RUN_MODEL, False):
                st.session_state[DF] = integra_deformacao(
                    deps=vc(incrementos), num_steps=steps, material=material
                )
                st.session_state[DF].to_csv("resultadoApp.csv")
                st.session_state[RUN_MODEL] = False

                if "meshes" in st.session_state:
                    st.session_state.pop("meshes")

            add_plots(right, st.session_state[DF], material)


########################################################################################
# %%          FUNCAO: ADICIONA PLOTS
########################################################################################


def add_plots(
    column: DeltaGenerator,
    df: DataFrame,
    material: Material,
    nP: int = 1000,
    nS: int = 1000,
):
    N: int = len(df)
    n = 0  # column.slider("Step", min_value=0, max_value=N - 1, value=0, step=1)

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
            fig.update_xaxes(
                row=i + 1, col=j + 1, title_text=x_labels[i, j], **axes_dict
            )
            fig.update_yaxes(
                row=i + 1, col=j + 1, title_text=y_labels[i, j], **axes_dict
            )

    ####################################################################################
    # %           INDICE DE VAZIOS POR P
    ####################################################################################

    fig.update_xaxes(
        row=1, col=1, type="log", range=[np.log10(np.min(df.p)), np.log10(np.max(df.p))]
    )
    fig.add_trace(
        row=1, col=1, trace=Scatter(x=df.p, y=df.e, showlegend=False, name="e vs lnP")
    )  # 0
    add_marker(fig, 1, 1, df.p[n], df.e[n])  # 1

    ####################################################################################
    # %           P x Q
    ####################################################################################

    p0max = np.max(df.s)
    pEnv = np.linspace(0, p0max, 100)
    qEnv = material.Mc * pEnv

    fig.add_trace(
        row=1, col=2, trace=Scatter(x=df.p, y=df.q, showlegend=False, name="Trajetoria")
    )  # 2
    fig.add_trace(
        row=1, col=2, trace=Scatter(x=pEnv, y=qEnv, showlegend=False, name="Envoltoria")
    )  # 3
    fig.update_xaxes(row=1, col=2, range=[0, p0max])
    fig.update_yaxes(row=1, col=2, range=[0, np.max(qEnv)])

    p0 = np.array(df.s).reshape(N, 1)
    p = p0 * np.linspace(0, 1, nP)
    q = np.array(material.q_plastic(p, p0))

    fig.add_trace(
        row=1,
        col=2,
        trace=Scatter(x=p[0], y=q[0], showlegend=False, name="Sup-Plast-Ini"),
    )  # 4
    fig.add_trace(
        row=1,
        col=2,
        trace=Scatter(x=p[n], y=q[n], showlegend=False, name="Sup. Plastif."),
    )  # 5

    add_marker(fig, 1, 2, df.p[n], df.q[n])  # 6

    ####################################################################################
    # %           TENSAO x DEFORMACAO
    ####################################################################################
    S1max = np.max(df.S1)
    fig.update_yaxes(row=2, col=1, range=[0, S1max])
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ex, y=df.Sx, name="Sx-Ex"))  # 7
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ex, y=df.Sy, name="Sy-Ex"))  # 8
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ex, y=df.Sz, name="Sz-Ex"))  # 9
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ey, y=df.Sx, name="Sx-Ey"))  # 10
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ey, y=df.Sy, name="Sy-Ey"))  # 11
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ey, y=df.Sz, name="Sz-Ey"))  # 12
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ez, y=df.Sx, name="Sx-Ez"))  # 13
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ez, y=df.Sy, name="Sy-Ez"))  # 14
    fig.add_trace(row=2, col=1, trace=Scatter(x=df.Ez, y=df.Sz, name="Sz-Ez"))  # 15

    add_marker(fig, 2, 1, df.Ex[n], df.Sx[n])  # 16
    add_marker(fig, 2, 1, df.Ex[n], df.Sy[n])  # 17
    add_marker(fig, 2, 1, df.Ex[n], df.Sz[n])  # 18
    add_marker(fig, 2, 1, df.Ey[n], df.Sx[n])  # 19
    add_marker(fig, 2, 1, df.Ey[n], df.Sy[n])  # 20
    add_marker(fig, 2, 1, df.Ey[n], df.Sz[n])  # 21
    add_marker(fig, 2, 1, df.Ez[n], df.Sx[n])  # 22
    add_marker(fig, 2, 1, df.Ez[n], df.Sy[n])  # 23
    add_marker(fig, 2, 1, df.Ez[n], df.Sz[n])  # 24

    ####################################################################################
    # %           SIGMA x TAU
    ####################################################################################

    p0 = np.array(df.s).reshape(N, 1)
    sig = p0 * np.linspace(0, 1, nP)
    M = np.divide(q, p, out=np.zeros_like(p), where=(p > 0))
    M[M > material.Mc] = material.Mc
    phi = np.asin(3 * M / (6 + M))
    mi = np.tan(phi)
    tau = mi * sig
    fig.add_trace(
        row=2,
        col=2,
        trace=Scatter(x=sig[0], y=tau[0], showlegend=False, name="Envoltoria"),
    )  # 25
    fig.add_trace(
        row=2,
        col=2,
        trace=Scatter(x=sig[n], y=tau[n], showlegend=False, name="Envoltoria"),
    )  # 26
    Tmax = np.max(tau[~np.isnan(tau)])
    Smax = np.max(sig[~np.isnan(sig)])
    fig.update_yaxes(row=2, col=2, range=[0, Tmax])
    fig.update_xaxes(row=2, col=2, range=[0, Smax])

    S1 = np.array(df.S1).reshape(N, 1)
    S2 = np.array(df.S2).reshape(N, 1)
    S3 = np.array(df.S3).reshape(N, 1)
    R12 = (S1 - S2) / 2
    R13 = (S1 - S3) / 2
    R23 = (S2 - S3) / 2
    C12 = (S1 + S2) / 2
    C13 = (S1 + S3) / 2
    C23 = (S2 + S3) / 2
    S12 = S2 + (S1 - S2) * np.linspace(0, 1, nS)
    S13 = S3 + (S1 - S3) * np.linspace(0, 1, nS)
    S23 = S3 + (S2 - S3) * np.linspace(0, 1, nS)
    T12 = np.sqrt(np.abs(R12**2 - (S12 - C12) ** 2))
    T13 = np.sqrt(np.abs(R13**2 - (S13 - C13) ** 2))
    T23 = np.sqrt(np.abs(R23**2 - (S23 - C23) ** 2))
    fig.add_trace(
        row=2, col=2, trace=Scatter(x=S12[n], y=T12[n], showlegend=True, name="Circ. 1")
    )  # 27
    fig.add_trace(
        row=2, col=2, trace=Scatter(x=S13[n], y=T13[n], showlegend=True, name="Circ. 2")
    )  # 28
    fig.add_trace(
        row=2, col=2, trace=Scatter(x=S23[n], y=T23[n], showlegend=True, name="Circ. 3")
    )  # 29

    add_marker(fig, 2, 2, S12[n][0], 0)  # 30
    add_marker(fig, 2, 2, S12[n][-1], 0)  # 31
    add_marker(fig, 2, 2, S13[n][0], 0)  # 32
    add_marker(fig, 2, 2, S13[n][-1], 0)  # 33
    add_marker(fig, 2, 2, S23[n][0], 0)  # 34
    add_marker(fig, 2, 2, S23[n][-1], 0)  # 35

    ####################################################################################
    # %           ADICIONA STEPS FIGURA 2D
    ####################################################################################
    steps = []
    for i in range(N):
        steps.append(
            {
                "label": str(i),
                "method": "restyle",
                "args": [
                    {
                        "x": [
                            [df.p[i]],  # 1
                            p[i],  # 5
                            [df.p[i]],  # 6
                            [df.Ex[i]],  # 16
                            [df.Ex[i]],  # 17
                            [df.Ex[i]],  # 18
                            [df.Ey[i]],  # 19
                            [df.Ey[i]],  # 20
                            [df.Ey[i]],  # 21
                            [df.Ez[i]],  # 22
                            [df.Ez[i]],  # 23
                            [df.Ez[i]],  # 24
                            sig[i],  # 26
                            S12[i],  # 27
                            S13[i],  # 28
                            S23[i],  # 29
                            [S12[i][0]],  # 30
                            [S12[i][-1]],  # 31
                            [S13[i][0]],  # 32
                            [S13[i][-1]],  # 33
                            [S23[i][0]],  # 34
                            [S23[i][-1]],  # 35
                        ],
                        "y": [
                            [df.e[i]],  # 1
                            q[i],  # 5
                            [df.q[i]],  # 6
                            [df.Sx[i]],  # 16
                            [df.Sy[i]],  # 17
                            [df.Sz[i]],  # 18
                            [df.Sx[i]],  # 19
                            [df.Sy[i]],  # 20
                            [df.Sz[i]],  # 21
                            [df.Sx[i]],  # 22
                            [df.Sy[i]],  # 23
                            [df.Sz[i]],  # 24
                            tau[i],  # 26
                            T12[i],  # 27
                            T13[i],  # 28
                            T23[i],  # 29
                            [0],  # 30
                            [0],  # 31
                            [0],  # 32
                            [0],  # 33
                            [0],  # 34
                            [0],  # 35
                        ],
                    },
                    [1, 5, 6] + list(range(16, 25)) + list(range(26, 36)),
                ],
            }
        )

    ####################################################################################
    # %           DEFINE SLIDER E FIGURA 2D
    ####################################################################################

    slider = [
        {
            "active": 0,
            "steps": steps,
            "currentvalue": {
                "visible": True,
                "prefix": "Step: ",
            },
        }
    ]

    fig.update_layout(sliders=slider, **fig_dict)

    tabs = column.tabs(["2D plot", "3D plot"])

    tabs[0].plotly_chart(fig, width="stretch", theme=None)

    ####################################################################################
    # %           PLOT 3D
    ####################################################################################

    fig3D = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.05,
        vertical_spacing=0.07,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
    )

    ####################################################################################
    # %           MALHA SUPERFICIE HW
    ####################################################################################

    nHW = 50

    smin = Smax / 2
    Smin = smin - smin * np.sqrt(3)
    Smax *= np.sqrt(3)
    s = np.linspace(Smin, Smax, nHW)
    if "meshes" in st.session_state:
        meshes = st.session_state["meshes"]
    else:

        s1, s2, s3 = np.meshgrid(s, s, s, indexing="ij")
        f = np.array(
            material.func_plastica(
                s1[..., None], s2[..., None], s3[..., None], p0.reshape(1, 1, 1, N)
            )
        )

        meshes = []
        for i in tqdm(range(f.shape[3])):
            verts, faces, _, _ = marching_cubes(f[..., i], level=0.0)
            verts[:, 0] = s[0] + verts[:, 0] * (s[-1] - s[0]) / (nHW - 1)
            verts[:, 1] = s[0] + verts[:, 1] * (s[-1] - s[0]) / (nHW - 1)
            verts[:, 2] = s[0] + verts[:, 2] * (s[-1] - s[0]) / (nHW - 1)
            meshes.append((verts, faces))
        st.session_state["meshes"] = meshes

    ####################################################################################
    # %           MALHA SUPERFICIE PxQxE
    ####################################################################################

    npqe = 50
    p0 = np.array(df.s).reshape(N, 1)
    p = np.linspace(0, np.max(p0), npqe)
    q = np.linspace(0, material.Mc * np.max(p0) / 2, npqe)
    e = np.linspace(np.min(df.e), np.max(df.e), npqe)

    if "meshesE" in st.session_state:
        meshesE = st.session_state["meshesE"]
    else:

        pp, ee, qq = np.meshgrid(p, e, q, indexing="ij")
        fe = np.array(
            material.func_plastica(pp[..., None], qq[..., None], ee[..., None])
        )

        meshesE = []
        for i in tqdm(range(fe.shape[3])):
            verts, faces, _, _ = marching_cubes(fe[..., i], level=0.0)
            verts[:, 0] = p[0] + verts[:, 0] * (p[-1] - p[0]) / (npqe - 1)
            verts[:, 1] = e[0] + verts[:, 1] * (e[-1] - e[0]) / (npqe - 1)
            verts[:, 2] = q[0] + verts[:, 2] * (q[-1] - q[0]) / (npqe - 1)
            meshesE.append((verts, faces))
        st.session_state["meshesE"] = meshesE

    ####################################################################################
    # %           SUPERFICIE INICIAL HW
    ####################################################################################

    verts, faces = meshes[0]

    fig3D.add_trace(
        row=1,
        col=1,
        trace=Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightblue",
            opacity=0.6,
            name="Superficie",
            showlegend=True,
        ),
    )  # 0

    ####################################################################################
    # %           PONTO INICIAL HW
    ####################################################################################

    fig3D.add_trace(
        Scatter3d(
            x=S2.ravel()[0:1],
            y=S3.ravel()[0:1],
            z=S1.ravel()[0:1],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Estado atual",
        )
    )  # 1

    ####################################################################################
    # %           EIXO HIDRO HW
    ####################################################################################

    fig3D.add_trace(
        Scatter3d(
            x=s,
            y=s,
            z=s,
            mode="lines",
            name="Eixo hidrostatico",
            line=dict(color="black"),
        ),
    )

    ####################################################################################
    # %           TRAJETORIA HW
    ####################################################################################

    fig3D.add_trace(
        Scatter3d(
            x=S2.ravel(),
            y=S3.ravel(),
            z=S1.ravel(),
            mode="lines",
            name="Trajetoria",
        )
    )

    ####################################################################################
    # %           SUPERFICIE INICIAL PxQxE
    ####################################################################################

    verts, faces = meshesE[0]

    fig3D.add_trace(
        row=1,
        col=2,
        trace=Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightblue",
            opacity=0.6,
            name="Superficie",
        ),
    )

    ####################################################################################
    # %           CURVA PE NO GRAFICO PxQxE
    ####################################################################################

    fig3D.add_trace(
        row=1,
        col=2,
        trace=Scatter3d(
            x=df.p,
            y=df.e,
            z=np.zeros(df.e.shape),
            showlegend=True,
            name="e vs lnP",
        ),
    )  # 0

    ####################################################################################
    # %           FRAMES GRAFICO 3D, STEPS E SLIDER
    ####################################################################################

    frames = []
    for i, (verts, faces) in enumerate(meshes):

        frames.append(
            Frame(
                name=str(i),
                data=[
                    Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                    ),
                    Scatter3d(
                        x=S2.ravel()[i : i + 1],
                        y=S3.ravel()[i : i + 1],
                        z=S1.ravel()[i : i + 1],
                        mode="markers",
                        marker=dict(size=6, color="red"),
                    ),
                ],
            )
        )

    fig3D.frames = frames

    # ------------------------------- Steps

    steps = []
    for frame in frames:
        step = {
            "label": frame.name,
            "method": "animate",
            "args": [[frame.name], {"mode": "immediate"}],
        }
        steps.append(step)

    # ------------------------------- slider

    slider = [{"steps": steps}]

    ####################################################################################
    # %           EIXOS HW
    ####################################################################################

    # Eixo X
    fig3D.add_trace(
        Scatter3d(
            x=[-50, Smax],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            name="Eixo X",
        )
    )

    # Eixo Y
    fig3D.add_trace(
        Scatter3d(
            x=[0, 0],
            y=[-50, Smax],
            z=[0, 0],
            mode="lines",
            name="Eixo Y",
        )
    )

    # Eixo Z
    fig3D.add_trace(
        Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-50, Smax],
            mode="lines",
            name="Eixo Z",
        )
    )

    ####################################################################################
    # %           FIGURA 3D FINAL
    ####################################################################################

    fig3D.update_layout(
        height=900,
        margin=dict(l=50, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(range=[Smin, Smax], title="σ<sub>2</sub>"),
            yaxis=dict(range=[Smin, Smax], title="σ<sub>3</sub>"),
            zaxis=dict(range=[Smin, Smax], title="σ<sub>1</sub>"),
        ),
        sliders=slider,
        scene_camera={
            "eye": {"x": 2.5, "y": 0.2, "z": 0.1},
        },
        scene2=dict(
            xaxis_title="p (kPa)",
            yaxis_title="e",
            zaxis_title="q (kPa)",
        ),
    )

    tabs[1].plotly_chart(fig3D, width="stretch", theme=None)


########################################################################################
# %%          EXECUCAO PRINCIPAL
########################################################################################

if __name__ == "__main__":
    setup_app(modelos)
