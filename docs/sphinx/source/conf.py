from typing import Literal, TypedDict


class PyDomainInfo(TypedDict):
    module: str
    fullname: str


import modelos

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Modelos constitutivos"
copyright = "2025, Diogo Rossi"
author = "Diogo Rossi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxnotes.comboroles",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
]

maximum_signature_line_length = 70

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = []

language = "pt-br"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = '<p style="text-align: center"><b>Modelos constitutivos</b></p>'
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
default_role = "code"


def linkcode_resolve(domain: Literal["py", "c", "cpp", "javascript"], info: PyDomainInfo):
    pass
