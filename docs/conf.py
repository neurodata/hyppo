# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

# General information about the project
year = datetime.date.today().year
project = "hyppo"
copyright = "2018-{}, NeuroData".format(year)
authors = "Sambit Panda"

# The short X.Y version
# Find hyppo version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "..", "hyppo", "__init__.py")):
    if line.startswith("__version__ = "):
        version = line.strip().split()[2][1:-1]

# The full version, including alpha/beta/rc tags
release = "alpha"

# -- Extension configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    # "numpydoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "super"
bibtex_default_style = "unsrt"

autodoc_typehints = "none"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    # objects without namespace
    "ndarray": "~numpy.ndarray",
}

# -- numpydoc
# Below is needed to prevent errors
numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# -- sphinx.ext.autosummary
autosummary_generate = True

# Otherwise, the Return parameter list looks different from the Parameters list
napoleon_use_rtype = False
# Otherwise, the Attributes parameter list looks different from the Parameters
# list
napoleon_use_ivar = True


# uncomment line 55
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": ["../examples", "../tutorials", "../sample_data", "../benchmarks"],
    # path where to save gallery generated examples
    "gallery_dirs": ["gallery", "tutorials", "sample_data", "benchmarks"],
    "filename_pattern": r"\.py",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "api/generated/backreferences",
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": "hyppo",
    # Insert links to documentation of objects in the examples
    "reference_url": {"hyppo": None},
}

html_css_files = [
    "css/custom.css",
]

# -- sphinx.ext.autodoc
# autoclass_content = "both"
# autodoc_default_flags = ["members", "inherited-members"]
# autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ["png"]

# -- sphinx options ----------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
master_doc = "index"
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_extra_path = []
modindex_common_prefix = ["hyppo."]
html_last_updated_fmt = "%b %d, %Y"
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

pygments_style = "sphinx"
smartquotes = False

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links_label": "Quick Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/neurodata/hyppo",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Paper",
            "url": "https://arxiv.org/abs/1907.02088",
            "icon": "fas fa-book",
        },
        {
            "name": "NeuroData",
            "url": "https://neurodata.io/",
            "icon": "fas fa-brain",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "neurodata",
    "github_repo": "hyppo",
    "github_version": "main",
    "doc_path": "docs",
    "galleries": sphinx_gallery_conf["gallery_dirs"],
    "gallery_dir": dict(
        zip(
            sphinx_gallery_conf["gallery_dirs"],
            sphinx_gallery_conf["examples_dirs"],
        )
    ),
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "hyppodoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "hyppo.tex", "hyppo Documentation", authors, "manual")]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "hyppo", "hyppo Documentation", [authors], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "hyppo",
        "hyppo Documentation",
        authors,
        "hyppo",
        "One line description of project.",
        "Miscellaneous",
    )
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Other -------------------------------------------------------------------


def setup(app):
    # to hide/show the prompt in code examples:
    app.add_js_file("js/copybutton.js")


# -- Accessibility -----------------------------------------------------------

language = "en"
extensions += ["sphinx_sitemap"]

html_baseurl = os.environ.get("SPHINX_HTML_BASE_URL", "https://127.0.0.1:8000/")
sitemap_locales = [None]
sitemap_url_scheme = "{link}"
