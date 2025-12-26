from typing import Literal
from dash import Dash
import webview
import os


def run(
    app: Dash,
    port: int = 8050,
    debug: bool = False,
    title: str = "My App",
    maximized: bool = True,
):

    def run_dash():
        app.run(debug=debug, port=port, use_reloader=False)

    webview.create_window(title, f"http://localhost:{port}/", maximized=maximized)
    webview.start(run_dash)
    os._exit(0)
