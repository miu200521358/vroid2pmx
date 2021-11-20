# -*- coding: utf-8 -*-
#

import wx
import time
from form.worker.BaseWorkerThread import BaseWorkerThread, task_takes_time


class LoadWorkerThread(BaseWorkerThread):

    def __init__(self, frame: wx.Frame, result_event: wx.Event, is_exec: bool, is_param: bool, is_param_advance: bool, is_param_bone: bool):
        self.elapsed_time = 0
        self.is_exec = is_exec
        self.is_param = is_param
        self.is_param_advance = is_param_advance
        self.is_param_bone = is_param_bone
        self.gauge_ctrl = frame.file_panel_ctrl.gauge_ctrl

        super().__init__(frame, result_event, frame.file_panel_ctrl.console_ctrl)

    @task_takes_time
    def thread_event(self):
        start = time.time()

        # 元モデルの読み込み
        self.result = self.frame.file_panel_ctrl.org_model_file_ctrl.load(is_check=False, is_sizing=False) and self.result

        self.elapsed_time = time.time() - start

    def thread_delete(self):
        pass

    def post_event(self):
        wx.PostEvent(self.frame, self.result_event(result=self.result and not self.is_killed, \
                                                   elapsed_time=self.elapsed_time, is_exec=self.is_exec, is_param=self.is_param, \
                                                   is_param_advance=self.is_param_advance, is_param_bone=self.is_param_bone))

