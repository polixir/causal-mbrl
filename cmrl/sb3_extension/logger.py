from typing import Dict, Any, Tuple, Union, List
from collections import defaultdict
import os

from stable_baselines3.common.logger import KVWriter, CSVOutputFormat, Logger, make_output_format


class MultiCSVOutputFormat(KVWriter):
    """
    Log to multi CSV format file, classified by key's prefix

    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.prefix_keys = []
        self.csv_output_formats = {}

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:
        key_values_dict = defaultdict(dict)
        key_excluded_dict = defaultdict(dict)

        for key in key_values:
            if "/" in key:
                assert len(key.split("/")) == 2
                prefix_key, real_key = key.split("/")
            else:
                prefix_key, real_key = "default", key
            key_values_dict[prefix_key][real_key] = key_values[key]
            key_excluded_dict[prefix_key][real_key] = key_excluded[key]

        for prefix_key in key_values_dict:
            if prefix_key not in self.prefix_keys:
                self.prefix_keys.append(prefix_key)
                self.csv_output_formats[prefix_key] = CSVOutputFormat(os.path.join(self.log_dir, f"{prefix_key}.csv"))

            self.csv_output_formats[prefix_key].write(key_values_dict[prefix_key],
                                                      key_excluded_dict[prefix_key])

    def close(self) -> None:
        """
        closes the file
        """
        for prefix_key in self.prefix_keys:
            self.csv_output_formats[prefix_key].close()


def configure(folder: str, format_strings: [List[str]]) -> Logger:
    """
    Configure the current logger.

    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    assert format_strings is not None

    format_strings = list(filter(None, format_strings))
    output_formats = []
    for f in format_strings:
        if f == "multi_csv":
            output_formats.append(MultiCSVOutputFormat(folder))
        else:
            output_formats.append(make_output_format(f, folder, log_suffix))

    logger = Logger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger


if __name__ == '__main__':
    from stable_baselines3.common.logger import Logger

    logger = Logger("1", [MultiCSVOutputFormat("./"), CSVOutputFormat("./test.csv")])

    logger.record("t/a", 1)
    logger.record("t/b", 2)
    logger.record("b/c", 3)
    logger.record("c/d", 4)
    logger.dump()

    logger.record("t/a", 1)
    logger.record("t/b", 2)
    logger.record("b/c", 3)
    logger.record("c/d", 4)
    logger.dump()
