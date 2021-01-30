import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="ucate",
    version="0.0.0",
    description="Exploring uncertainty for CATE inference",
    long_description_content_type="text/markdown",
    url="https://github.com/OATML/ucate",
    author="Andrew Jesson",
    author_email="andrew.jesson@cs.ox.ac.uk",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "rpy2",
        "click",
        "numpy>=1.16.0,<1.19.0",
        "scipy",
        "pandas",
        "sklearn",
        "seaborn",
        "matplotlib",
        "tensorflow==2.3.1",
        "tensorflow-probability==0.11.1",
    ],
    entry_points={
        "console_scripts": ["ucate=ucate.application.main:cli"],
    },
)
