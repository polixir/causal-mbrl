import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from torch.utils.tensorboard import SummaryWriter

from cmrl.util.logger import AverageMeter, Logger, MetersGroup


class TestAverageMeter(TestCase):
    def setUp(self) -> None:
        self.meter = AverageMeter()

    def test_update(self):
        for i in range(10):
            self.meter.update(i)
        assert self.meter.value() == 4.5


class TestMetersGroup(TestCase):
    def setUp(self) -> None:
        tempdir = Path(tempfile.gettempdir())
        self.log_dir = tempdir / "temp_log"
        self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))

        self.meter_group = MetersGroup(
            file_path=self.log_dir / "tests",
            formatting=[
                ("column0", "C0", "int"),
                ("column1", "C1", "float"),
                ("column2", "C2", "float"),
            ],
            tensorboard_writer=self._tb_writer,
            disable_console_dump=False,
            tb_index_key="column0",
        )

        values = [
            [9, 8, 7, 6, 5, 4, 3, 2, 1],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [11, 22, 33, 44, 55, 66, 77, 88, 99],
        ]
        for i in range(len(values[0])):
            for j in range(len(values)):
                key = "column{}".format(j)
                self.meter_group.log(key, values[j][i])
            self.meter_group.dump(i, "tests")

    def test_csv(self):
        pass

    def test_tb(self):
        pass


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        tempdir = Path(tempfile.gettempdir())
        self.log_dir = tempdir / "temp_logger"
        self.logger = Logger(self.log_dir)
        self.logger.register_group(
            group_name="test0",
            log_format=[
                ("column0", "C0", "int"),
                ("column1", "C1", "float"),
                ("column2", "C2", "float"),
            ],
            tb_index_key="column0",
        )
        self.logger.register_group(
            group_name="test1",
            log_format=[
                ("column0", "C0", "int"),
                ("column1", "C1", "float"),
                ("column2", "C2", "float"),
            ],
        )
        self.values0 = [
            [9, 8, 7, 6, 5, 4, 3, 2, 1],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [11, 22, 33, 44, 55, 66, 77, 88, 99],
        ]

        self.values1 = [
            [9, 8, 7, 6, 5, 4, 3, 2, 1],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [11, 22, 33, 44, 55, 66, 77, 88, 99],
        ]

    def test_log_data(self):
        for i in range(len(self.values0[0])):
            self.logger.log_data(
                "test0",
                dict(
                    [
                        ("column{}".format(j), self.values0[j][i])
                        for j in range(len(self.values0))
                    ]
                ),
            )
            self.logger.log_data(
                "test1",
                dict(
                    [
                        ("column{}".format(j), self.values1[j][i])
                        for j in range(len(self.values0))
                    ]
                ),
            )
