# -*- coding: utf-8 -*-
#

from mmd.PmxData import PmxModel
from utils.MLogger import MLogger # noqa

logger = MLogger(__name__)


class MExportOptions:

    def __init__(self, version_name: str, logging_level: int, max_workers: int, vrm_model: PmxModel, physics_flg: bool, output_path: str, \
                 param_options: dict, monitor, is_file: bool, outout_datetime: str):
        self.version_name = version_name
        self.logging_level = logging_level
        self.vrm_model = vrm_model
        self.physics_flg = physics_flg
        self.output_path = output_path
        self.param_options = param_options
        self.monitor = monitor
        self.is_file = is_file
        self.outout_datetime = outout_datetime
        self.max_workers = max_workers


class MOptionsDataSet:
    def __init__(self):
        pass
