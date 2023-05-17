"""

Author: F. Thomas
Date: July 19, 2021

"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

required = [
    "numpy",
    "scipy>=1.9",
    "matplotlib",
    "uproot"
]

setuptools.setup(
    name="cresana",
    version="0.2.0",
    author="Florian Thomas",
    author_email="fthomas@uni-mainz.de",
    description="https://github.com/MCFlowMace/CRESana",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MCFlowMace/CRESana",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=required)
