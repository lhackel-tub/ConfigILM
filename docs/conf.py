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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# -- Project information -----------------------------------------------------

project = "ConfigVLM"
copyright = "2023, Leonard Wayne Hackel"
author = "Leonard Wayne Hackel"
html_title = project

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    # Other popular choices:
    # "sphinx_design",
    # "sphinx_cli_recorder",
    # "sphinxcontrib.mermaid",
    # "sphinx_design",
    "sphinxcontrib.bibtex",
    # "sphinx_comments",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    # "furo_myst_nb_css_fixes",
    # "sphinx_external_toc",
]

# Only useful if sub-urls are accessed
# extlinks = {
# }

comments_config = {"hypothesis": True}

bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "plain"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    # "amsmath",
    # "html_admonition",
    # "html_image",
    # "strikethrough",
    # "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_substitutions = {
    "ben": "[BigEarthNet](https://bigearth.net/)",
    "timm": "[PyTorch Image Models]"
    "(https://github.com/rwightman/pytorch-image-models/)",
    "issues": "[GitHub issues](https://github.com/lhackel-tub/ConfigVLM/issues)",
    "lmdb": "[LMDB](http://www.lmdb.tech/doc/)",
    "bendocs": "[BigEarthNet Guide](https://github.com/kai-tub/ben-docs)",
}

nb_custom_formats = {
    ".ipynb": [
        "common_nb_preprocessors.myst_nb_metadata_injector",
        {
            "prefix": "#",
            "delimiter": "=",
            "extra_tags": ["scroll-output", "only-light", "only-dark"],
        },
    ]
}

# always fail CI pipeline when nb cannot be executed
nb_execution_raise_on_error = True

# show line numbers in source code
nb_number_source_lines = False

# Recommendation from furo
# https://pradyunsg.me/furo/kitchen-sink/api/
autodoc_typehints = "description"
autodoc_class_signature = "separated"

source_suffix = {".ipynb": "myst-nb", ".md": "myst-nb"}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "md_representations",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "furo"

html_static_path = ["_static"]

html_theme_options = {
    "footer_icons": [
        {
            "name": "BIFOLD",
            "url": "https://bifold.berlin/",
            "html": """
                <img src="/_static/BIFOLD_Logo_farbig.svg" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo">
            """,
            "class": "",
        },
        {
            "name": "RSiM",
            "url": "https://rsim.berlin/",
            # Yes, you shouldn't add style inline in HTML but just think of it as a pre-modern
            # version of tailwindcss, as this seems to be better for whatever reason ;)
            "html": """
                <img src="/_static/RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo">
            """,
            "class": "",
        },
    ],
}
