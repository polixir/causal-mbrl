![](/img/cmrl_logo.png)

# causal-mbrl

<a href="https://github.com/FrankTianTT/causal-mbrl"><img src="https://github.com/FrankTianTT/causal-mbrl/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/FrankTianTT/causal-mbrl"><img src="https://codecov.io/github/FrankTianTT/causal-mbrl/branch/main/graph/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/FrankTianTT/causal-mbrl/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>

Toolkit of Causal Model-based Reinforcement learning.

# Installation and Usage

```shell
# create conda env
conda create -n cmrl python=3.8
conda activate cmrl
# install pytorch by conda if there is not cuda in your env
conda install pytorch cudatoolkit=11.3 -c pytorch
# install cmrl and its dependent packages
pip install -e .
```
