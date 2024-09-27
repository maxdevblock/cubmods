# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from dataclasses import dataclass, field

import sphinxcontrib.bibtex.plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join('..',)))


# -- Project information -----------------------------------------------------

project = 'CUBmods'
copyright = '2024, Massimo Pierini'
author = 'Massimo Pierini'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    #'sphinx.ext.napoleon',
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    #"sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    #"sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    'numpydoc', # Alternative to napoleon
    'sphinxcontrib.bibtex', # for citation and bibliography
]

# bibliography
bibtex_bibfiles = [
    'cub.bib'
]
latex_elements = {
    'preamble': r'''
    \usepackage{upgreek}
    '''
}

# https://github.com/mcmtroffaes/sphinxcontrib-bibtex/blob/develop/test/roots/test-citation_style_round_brackets/conf.py
def bracket_style() -> BracketStyle:
    return BracketStyle(
        left="(",
        right=")",
    )

@dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
    bracket_textual: BracketStyle = field(default_factory=bracket_style)
    bracket_author: BracketStyle = field(default_factory=bracket_style)
    bracket_label: BracketStyle = field(default_factory=bracket_style)
    bracket_year: BracketStyle = field(default_factory=bracket_style)

sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", MyReferenceStyle
)

#extensions = ["sphinxcontrib.bibtex"]
exclude_patterns = ["_build"]
#bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year_round"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = [
# ]
# autosummary_mock_imports = [
#     'cubmods.cubsh2',
#     'cubmods.cubsh2_yxwxu',
#     'cubmods.cushk',
# ]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']