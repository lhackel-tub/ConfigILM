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
                # <img src="_static/BIFOLD_Logo_farbig.svg" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo">
            "html": """
            <svg version="1.1" id="Ebene_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
            	 viewBox="0 0 932.3 187" style="enable-background:new 0 0 932.3 187; display: block; height: 2em; width: auto; margin-right: 1em;" xml:space="preserve">
            <style type="text/css">
            	.st0{fill:#7FCCE0;}
            	.st1{fill:#002D62;}
            	.st2{fill:#B1DEEB;}
            	.st3{fill:#2ABAD4;}
            </style>
            <polygon class="st0" points="233.7,67.5 166.2,0 150.5,140.5 338,171.8 "/>
            <polygon class="st1" points="348,60.6 240.6,60.6 348,167.9 "/>
            <polygon class="st2" points="166.2,0 0,73.6 150.5,140.5 "/>
            <polygon class="st3" points="150.5,140.5 253.2,187 338,171.8 "/>
            <g>
            	<path class="st1" d="M881.9,80.6H862V148h18.8c8,0,14.3-2.8,19-8.5c4.6-5.7,7-14.3,7-25.9c0-10.2-2.1-18.3-6.3-24.2
            		C896.2,83.5,890.1,80.6,881.9,80.6 M881.9,167.9h-44.5V60.5h44.9c10.7,0,19.9,2.3,27.4,6.8c7.5,4.5,13.2,10.8,17,18.7
            		c3.8,8,5.7,17.1,5.7,27.6c0,11.1-2.1,20.7-6.2,28.8c-4.1,8.2-10,14.5-17.5,18.9C901.1,165.7,892.2,167.9,881.9,167.9 M822.9,167.9
            		h-74.1V60.5h24.6v88.1H820L822.9,167.9z M648,114.5c0,10.3,2.6,18.7,7.7,25.2c5.1,6.5,12,9.7,20.8,9.7c5.4,0,10.2-1.3,14.6-4
            		c4.4-2.7,7.8-6.7,10.3-12.1c2.5-5.4,3.8-12,3.8-19.7c0-6.6-1.2-12.6-3.5-17.8c-2.3-5.3-5.6-9.4-9.9-12.4c-4.3-3-9.2-4.5-14.9-4.5
            		c-5.3,0-10.1,1.5-14.5,4.4c-4.4,2.9-7.9,7.1-10.5,12.6C649.3,101.2,648,107.5,648,114.5 M676.6,58.1c11.2,0,20.8,2.4,29,7.3
            		c8.2,4.9,14.4,11.6,18.7,20c4.3,8.4,6.5,17.8,6.5,28.1c0,11-2.3,20.7-6.9,29.2c-4.6,8.5-11,15.2-19.2,20
            		c-8.2,4.8-17.6,7.2-28.1,7.3c-11.1,0-20.7-2.4-28.8-7.3c-8.2-4.9-14.4-11.6-18.8-20c-4.4-8.4-6.6-17.8-6.6-28.1
            		c0-10.8,2.3-20.5,6.9-29.1c4.6-8.5,11-15.2,19.2-20.1C656.7,60.7,666.1,58.3,676.6,58.1 M606.8,125h-36.2v43H546V60.7h65.4
            		l2.8,19.3h-43.6v25.6h33.3L606.8,125z M495.3,60.5h24.6v107.4h-24.6V60.5z M437.1,122.4h-17.8v25.7h17.2c5.1,0,9.1-1.1,12.2-3.2
            		c3.1-2.2,4.6-5.2,4.6-9.1c0-4.2-1.4-7.5-4.3-9.9C446,123.6,442.1,122.4,437.1,122.4 M432.2,80.4h-13v22.3H430
            		c5.4,0,9.8-0.9,13.2-2.7c3.4-1.8,5.1-4.5,5.1-8.1C448.3,84.3,443,80.4,432.2,80.4 M437,167.9h-42.3V60.5h39.6
            		c11.9,0,21.5,2.5,28.8,7.5c7.3,5,11,12.7,11,23c0,6.5-3.6,13-10.7,19.3c4.7,2.9,8.5,6.6,11.3,10.9c2.8,4.3,4.2,9.3,4.2,14.8
            		c0,10.6-3.9,18.6-11.7,23.9C459.3,165.3,449.3,167.9,437,167.9"/>
            </g>
            </svg>
            """,
            "class": "",
        },
        {
            "name": "RSiM",
            "url": "https://rsim.berlin/",
            # Yes, you shouldn't add style inline in HTML but just think of it as a pre-modern
            # version of tailwindcss, as this seems to be better for whatever reason ;)
            # <!--<img src="_static/RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo"> -->
            "html": """
            <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
             width="987.000000pt" height="337.000000pt" viewBox="0 0 987.000000 337.000000"
             preserveAspectRatio="xMidYMid meet" style="display: block; height: 2em; width: auto">
            <g transform="translate(0.000000,337.000000) scale(0.100000,-0.100000)"
            fill="#070685" stroke="none">
            <path d="M3555 3350 c-149 -15 -344 -74 -454 -138 -293 -169 -426 -437 -392
            -788 31 -317 172 -523 475 -696 77 -43 284 -138 336 -153 27 -8 99 -35 245
            -92 128 -50 293 -133 357 -181 81 -60 142 -133 174 -209 26 -64 28 -77 28
            -208 1 -133 -1 -143 -29 -209 -43 -99 -144 -202 -247 -250 -263 -124 -663
            -122 -1133 3 -44 12 -89 26 -100 31 -11 5 -42 16 -69 25 -26 8 -57 20 -67 26
            -18 9 -19 3 -19 -174 l0 -183 43 -21 c62 -30 206 -70 342 -95 102 -18 167 -22
            430 -25 l310 -4 140 30 c277 58 462 159 608 330 72 84 128 194 157 308 21 81
            29 235 19 342 -30 314 -190 514 -554 693 -94 46 -313 138 -330 138 -20 0 -310
            122 -399 168 -88 46 -126 73 -191 137 -68 66 -87 92 -111 150 -27 66 -28 78
            -29 215 l0 145 37 71 c66 128 184 213 356 257 174 44 513 24 737 -44 62 -18
            233 -76 276 -94 13 -5 26 24 72 154 32 88 57 163 57 168 0 22 -283 108 -474
            144 -175 32 -446 45 -601 29z"/>
            <path d="M5407 3324 c-159 -43 -325 -149 -417 -265 -35 -46 -26 -41 50 24 58
            50 149 101 241 134 69 25 86 27 234 28 152 0 164 -2 240 -29 94 -33 192 -89
            254 -144 l44 -38 -84 -51 c-144 -86 -213 -121 -229 -115 -8 3 -37 15 -65 28
            -94 42 -167 74 -171 74 -2 0 -4 -82 -4 -182 l0 -183 -58 -24 c-31 -13 -91 -39
            -133 -57 -42 -19 -79 -34 -83 -34 -3 0 -6 73 -6 163 l0 162 33 19 c17 11 49
            27 69 37 38 19 57 39 39 39 -6 0 -11 -4 -11 -10 0 -5 -5 -10 -10 -10 -6 0 -33
            -11 -62 -25 -54 -26 -74 -30 -80 -17 -1 5 -39 28 -83 52 -139 75 -212 144
            -191 179 4 6 22 13 41 17 19 3 33 10 30 14 -6 10 -99 22 -118 14 -64 -24 12
            -142 202 -310 l106 -94 6 57 c5 46 6 39 5 -35 l-1 -94 -202 -111 c-111 -62
            -201 -116 -200 -121 2 -5 43 -26 92 -48 106 -46 100 -47 305 32 l135 52 50
            -15 c28 -9 70 -23 95 -33 45 -17 46 -17 110 8 36 14 76 29 90 33 24 7 24 7 5
            -8 -19 -14 -11 -20 145 -95 91 -43 172 -82 180 -85 8 -3 53 -22 100 -42 51
            -21 110 -38 147 -42 59 -6 63 -4 92 25 34 33 38 61 19 123 -14 50 -28 50 -28
            0 0 -43 -20 -71 -51 -71 -19 0 -19 3 9 53 57 100 83 197 85 315 1 59 -2 135
            -6 168 l-8 61 33 14 c18 7 36 18 40 23 4 6 8 84 8 173 l0 162 -37 -14 c-21 -7
            -78 -31 -128 -53 l-90 -40 -48 40 c-110 92 -198 138 -335 173 -135 34 -270 34
            -395 -1z m968 -226 c3 -24 5 -77 3 -118 l-3 -74 -200 -81 c-110 -45 -216 -88
            -235 -97 -19 -8 -53 -21 -75 -28 -22 -7 -44 -16 -50 -20 -5 -3 -32 -15 -59
            -25 -27 -10 -61 -24 -75 -31 -14 -8 -29 -10 -34 -5 -4 4 -7 46 -5 92 l3 84 75
            37 c93 46 330 159 470 224 58 26 119 56 135 66 17 10 33 18 37 18 4 0 10 -19
            13 -42z m-137 -361 c33 -88 47 -224 33 -325 -13 -97 -28 -152 -41 -152 -35 0
            -219 119 -332 215 -72 60 -78 63 -78 40 0 -14 -4 -25 -9 -25 -5 0 -18 -3 -28
            -6 -17 -6 -17 -6 1 8 14 11 20 33 25 94 l6 79 195 66 c107 36 198 64 202 62 4
            -2 15 -27 26 -56z m-492 -272 c-11 -8 -27 -14 -35 -14 -13 1 -12 2 2 6 9 2 17
            9 17 14 0 5 8 9 18 9 15 -1 15 -2 -2 -15z"/>
            <path d="M30 1681 l0 -1641 200 0 200 0 0 680 0 680 334 0 335 0 44 -77 c67
            -118 148 -258 207 -358 29 -49 77 -130 105 -180 29 -49 99 -171 157 -270 58
            -99 143 -245 190 -325 l84 -145 232 -3 c128 -1 232 1 232 4 0 4 -21 43 -47 86
            -27 44 -62 102 -79 131 -17 29 -88 146 -159 262 -70 116 -142 235 -160 265
            -17 30 -89 149 -160 265 -71 116 -141 233 -157 260 -16 28 -47 79 -68 114 -42
            66 -49 91 -26 91 25 0 171 72 246 122 178 118 285 263 347 469 26 89 28 105
            28 284 0 167 -3 200 -23 275 -87 320 -291 508 -650 596 -180 44 -225 46 -829
            51 l-583 5 0 -1641z m1219 1260 c327 -76 461 -243 461 -576 0 -323 -141 -510
            -445 -593 -86 -23 -107 -24 -462 -29 l-373 -5 0 617 0 617 363 -5 c327 -4 371
            -6 456 -26z"/>
            <path d="M6620 1680 l0 -1630 175 0 175 0 0 1183 c0 1051 -7 1530 -26 1665 -6
            48 9 57 24 15 6 -16 38 -100 72 -188 34 -88 70 -182 80 -210 43 -116 122 -322
            134 -350 7 -16 18 -43 24 -60 10 -29 77 -202 127 -330 13 -33 42 -109 65 -170
            82 -217 91 -240 101 -265 6 -14 34 -86 61 -160 62 -163 79 -209 158 -415 34
            -88 73 -189 87 -225 13 -36 35 -92 48 -125 13 -33 35 -91 49 -130 15 -38 31
            -79 36 -90 5 -11 19 -48 31 -82 l22 -63 153 0 c136 0 154 2 163 18 5 9 55 136
            111 282 55 146 107 279 115 295 7 17 16 41 20 55 4 14 13 39 20 55 8 17 30 73
            50 125 78 204 107 276 115 295 5 11 20 49 33 85 13 36 59 155 101 265 43 110
            89 229 102 265 14 36 31 79 39 95 7 17 16 41 20 55 4 14 17 50 30 80 12 30 34
            85 49 123 14 37 37 95 50 130 14 34 41 105 61 157 20 52 50 131 67 175 17 44
            36 94 43 110 7 17 30 74 51 128 20 54 40 94 44 90 4 -4 3 -48 -2 -98 -5 -49
            -12 -696 -14 -1437 l-5 -1348 188 0 188 0 0 1630 0 1630 -286 0 -286 0 -12
            -42 c-7 -24 -19 -56 -26 -73 -8 -16 -44 -109 -80 -205 -37 -96 -73 -188 -81
            -205 -7 -16 -22 -55 -33 -85 -11 -30 -58 -154 -106 -275 -48 -121 -95 -245
            -106 -275 -10 -30 -23 -64 -28 -75 -6 -11 -26 -63 -46 -115 -90 -233 -107
            -277 -120 -305 -7 -16 -16 -41 -20 -55 -4 -14 -13 -38 -20 -55 -8 -16 -29 -70
            -48 -120 -19 -49 -57 -148 -85 -220 -50 -129 -105 -271 -184 -475 -23 -58 -43
            -107 -45 -109 -11 -14 -29 26 -103 224 -54 145 -122 324 -134 350 -8 17 -17
            41 -21 55 -4 14 -13 39 -20 55 -15 34 -52 130 -150 385 -65 170 -87 228 -141
            368 -34 88 -64 168 -104 272 -45 117 -117 303 -145 375 -56 142 -167 433 -195
            512 -37 103 -2 93 -325 93 l-285 0 0 -1630z"/>
            <path d="M5420 1080 l0 -1030 190 0 190 0 0 1030 0 1030 -190 0 -190 0 0
            -1030z"/>
            </g>
            </svg>
            """,
            "class": "",
        },
    ],
}
