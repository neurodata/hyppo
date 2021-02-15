import os
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install

PACKAGE_NAME = "hyppo"
DESCRIPTION = "A comprehensive independence testing package"
with open("README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = ("Sambit Panda",)
AUTHOR_EMAIL = "spanda3@jhu.edu"
URL = "https://github.com/neurodata/hyppo"
MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.6
REQUIRED_PACKAGES = [
    "numpy>=1.17",
    "scipy>=1.4.0",
    "numba>=0.46",
    "scikit-learn>=0.19.1",
]

# Find mgc version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "hyppo", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")
        version = "v{}".format(VERSION)

        if tag != version:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, version
            )
            sys.exit(info)


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    include_package_data=True,
    test_suite="tests",
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
