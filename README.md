# causal-mbrl

![](/img/cmrl_logo.png)

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
