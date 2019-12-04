import os
import sys
from setuptools import setup, find_packages
from sys import platform

PACKAGE_NAME = "mgc"
DESCRIPTION = "A comprehensive independence testing package"
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = ("Sambit Panda",)
AUTHOR_EMAIL = "spanda3@jhu.edu"
URL = "https://github.com/sampan501/mgc"
MINIMUM_PYTHON_VERSION = 3, 5  # Minimum of Python 3.5
REQUIRED_PACKAGES = [
    "numpy>=1.17",
    "scipy>=1.3.1",
    "numba>=0.46",
    "scikit-learn>=0.19.1",
]

# Find mgc version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "mgc", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    test_suite="tests",
)