# -*- coding: utf-8 -*-
#

import logging
import re
import os
import wx
import time
import gc
from pathlib import Path

from form.worker.BaseWorkerThread import BaseWorkerThread, task_takes_time
from service.VroidExportService import VroidExportService
from module.MOptions import MExportOptions
from utils.MLogger import MLogger # noqa

logger = MLogger(__name__)


class ExportWorkerThread(BaseWorkerThread):

    def __init__(self, frame: wx.Frame, result_event: wx.Event, is_exec_saving: bool, is_out_log: bool):
        self.elapsed_time = 0
        self.frame = frame
        self.result_event = result_event
        self.gauge_ctrl = frame.file_panel_ctrl.gauge_ctrl
        self.is_exec_saving = is_exec_saving
        self.is_out_log = is_out_log
        self.options = None

        super().__init__(frame, self.result_event, frame.file_panel_ctrl.console_ctrl)

    @task_takes_time
    def thread_event(self):
        try:
            start = time.time()

            self.result = self.frame.file_panel_ctrl.org_model_file_ctrl.load() and self.result

            if self.result:
                self.options = MExportOptions(\
                    version_name=self.frame.version_name, \
                    logging_level=self.frame.logging_level, \
                    vrm_model=self.frame.file_panel_ctrl.org_model_file_ctrl.data.copy(), \
                    physics_flg=None, \
                    output_path=self.frame.file_panel_ctrl.output_pmx_file_ctrl.file_ctrl.GetPath(), \
                    param_options={}, \
                    monitor=self.frame.file_panel_ctrl.console_ctrl, \
                    is_file=False, \
                    outout_datetime=logger.outout_datetime, \
                    max_workers=(1 if self.is_exec_saving else min(5, 32, os.cpu_count() + 4)))

                self.result = VroidExportService(self.options).execute() and self.result

            self.elapsed_time = time.time() - start
        except Exception as e:
            logger.critical("Vroid2Pmx処理が意図せぬエラーで終了しました。", e, decoration=MLogger.DECORATION_BOX)
        finally:
            try:
                logger.debug("★★★result: %s, is_killed: %s", self.result, self.is_killed)
                if self.is_out_log or (not self.result and not self.is_killed):
                    # ログパス生成
                    output_pmx_path = self.frame.file_panel_ctrl.output_pmx_file_ctrl.file_ctrl.GetPath()
                    output_log_path = re.sub(r'\.pmx$', '.log', output_pmx_path)

                    # 出力されたメッセージを全部出力
                    self.frame.file_panel_ctrl.console_ctrl.SaveFile(filename=output_log_path)
            except Exception:
                pass

            logging.shutdown()

    def thread_delete(self):
        del self.options
        gc.collect()
        
    def post_event(self):
        wx.PostEvent(self.frame, self.result_event(result=self.result and not self.is_killed, elapsed_time=self.elapsed_time))
