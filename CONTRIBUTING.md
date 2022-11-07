# Contributing

In order to contibute to this repository you will need developer access to this repo.
To know more about the project go to the [README](README.md) first.

## Pre-commit hooks

Pre-commits hooks have been configured for this project using the
[pre-commit](https://pre-commit.com/) library:

- [black](https://github.com/psf/black) python formatter
- [flake8](https://flake8.pycqa.org/en/latest/) python linter

To get them going on your side, first install pre-commit:

```bash
pip install pre-commit
```

Then run the following commands from the root directory of this repository:

```bash
pre-commit install
pre-commit run --all-files
```

These pre-commits are applied to all the files, except the directory tmp/
(see .pre-commit-config.yaml)

## Git Commit Messages

Commits should start with a Capital letter and should be written in present tense (e.g. __:tada: Add cool new feature__
instead of __:tada: Added cool new feature__).
You should also start your commit message with **one** applicable emoji. This does not only look great but also makes
you rethink what to add to a commit. Make many but small commits!

Emoji | Description
------|------------
:tada: `:tada: ` | When you added a cool new feature.
:wrench: `:wrench:` | When you refactored / improved a small piece of code.
:hammer: `:hammer:` | When you refactored / improved large parts of the code.
:sparkles: `:sparkles:` | When you applied clang-format.
:art: `:art:` | When you improved / added assets like themes.
:rocket: `:rocket:` | When you improved performance.
:memo: `:memo:` | When you wrote documentation.
:bug: `:bug:` | When you fixed a bug.
:twisted_rightwards_arrows: `:twisted_rightwards_arrows:` | When you merged a branch.
:fire: `:fire:` | When you removed something.
:truck: `:truck:` | When you moved / renamed something.
:white_check_mark:  `:white_check_mark:`| When you  add a new unit test
