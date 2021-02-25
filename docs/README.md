# Building Docs

We currently use Sphinx

If you only want to get the documentation, this can be found at [https://hyppo.neurodata.io](https://hyppo.neurodata.io).

## Python Dependencies

You will need to install all the dependencies as defined in `requirements.txt` file. The above can be installed by entering:

    pip3 install -r requirements.txt

in the `docs/` directory. This is of course in addition to the package requirements. Here are the documentation requirements:

    sphinxcontrib-rawfiles==0.1.1
    nbsphinx==0.8.0
    ipython==7.19.0
    ipykernel==5.4.3
    sphinx==3.3
    sphinx_rtd_theme==0.4.3
    sphinx-gallery==0.8.2
    numpydoc==0.7.0
    recommonmark
    matplotlib
    seaborn

and make sure that you install the package dependencies as well:

    numpy>=1.17
    scipy>=1.4.0
    numba>=0.46
    scikit-learn>=0.22
    joblib>=0.17.0

## Pandoc dependency

In addition, you need to install `pandoc` for `nbsphinx`. If you are on linux, you can enter:

    sudo apt-get install pandoc

If you are on macOS and have `homebrew` installed, you can enter:

    brew install pandoc

Otherwise, you can visit [pandoc installing page](https://pandoc.org/installing.html) for more information.

## Generating the documentation

To build the HTML documentation, enter:

    make html

in the `docs/` directory. If all goes well, this will generate a `build/html/` subdirectory containing the built documentation.
