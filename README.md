![](/img/cmrl_logo.png)


# causal-mbrl

<a><img src="https://github.com/FrankTianTT/causal-mbrl/actions/workflows/python-app.yml/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Toolkit of Causal-model-based Reinforcement learning.

# Install

```shell
# create conda env
conda create -n cmrl python=3.8
conda activate cmrl
# install pytorch
conda install pytorch cudatoolkit=11.3 -c pytorch
# install stable-baselines3
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# install this package
pip install -e .
```
