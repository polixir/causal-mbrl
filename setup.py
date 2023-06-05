from pathlib import Path

from setuptools import find_packages, setup

def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

with open("README.md", "r") as f:
    long_description = f.read()


init_str = Path("cmrl/__init__.py").read_text()
version = init_str.split("__version__ = ")[1].rstrip().strip('"')

setup(
    name="cmrl",
    version=version,
    author="Honglong Tian",
    description="A Toolkit of Causal-model-based Reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrankTianTT/causal-mbrl",
    packages=[package for package in find_packages() if package.startswith("cmrl")],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=reqs_main,
    extras_require={
        "dev": reqs_main + reqs_dev,
    },
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
