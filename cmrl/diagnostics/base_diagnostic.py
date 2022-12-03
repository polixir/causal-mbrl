from typing import Union
import pathlib


class BaseDiagnostic:
    def __init__(self, exp_dir: Union[str, pathlib.Path]):
        if isinstance(exp_dir, str):
            self.exp_dir = pathlib.Path(exp_dir)
        else:
            self.exp_dir = exp_dir
        pass
