# -*- coding: utf-8 -*-
#

import os
import wx
import sys
import argparse
import numpy as np
import multiprocessing

from form.MainFrame import MainFrame
from utils.MLogger import MLogger
from utils import MFileUtils

VERSION_NAME = "2.01.01_β01"

# 指数表記なし、有効小数点桁数6、30を超えると省略あり、一行の文字数200
np.set_printoptions(suppress=True, precision=6, threshold=30, linewidth=200)

# Windowsマルチプロセス対策
multiprocessing.freeze_support()

if __name__ == "__main__":
    mydir_path = MFileUtils.get_mydir_path(sys.argv[0])

    if len(sys.argv) > 3 and "--motion_path" in sys.argv:
        if os.name == "nt":
            import winsound  # Windows版のみインポート

        # 終了音を鳴らす
        if os.name == "nt":
            # Windows
            try:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
            except Exception:
                pass
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", default=20, type=int)
        parser.add_argument("--log_mode", default=0, type=int)
        parser.add_argument("--out_log", default=0, type=int)
        parser.add_argument("--is_saving", default=1, type=int)
        parser.add_argument("--lang", default="ja_JP", type=str)
        args = parser.parse_args()

        # ロギングレベル
        is_out_log = True if args.out_log == 1 else False
        # 省エネモード
        is_saving = True if args.is_saving == 1 else False

        MLogger.initialize(level=args.verbose, is_file=False, target_lang=args.lang, mode=args.log_mode)
        logger = MLogger(__name__)

        log_level_name = ""
        if args.verbose == MLogger.FULL:
            # フルデータの場合
            log_level_name = logger.transtext("（全打ち版）")
        elif args.verbose == MLogger.DEBUG_FULL:
            # フルデータの場合
            log_level_name = logger.transtext("（全打ちデバッグ版）")
        elif args.verbose == MLogger.DEBUG:
            # テスト（デバッグ版）の場合
            log_level_name = logger.transtext("（デバッグ版）")
        elif args.verbose == MLogger.TIMER:
            # 時間計測の場合
            log_level_name = logger.transtext("（タイマー版）")
        elif not is_saving:
            # 省エネOFFの場合
            log_level_name = logger.transtext("（ハイスペック版）")
        elif is_out_log:
            # ログありの場合
            log_level_name = logger.transtext("（ログあり版）")

        now_version_name = "{0}{1}".format(VERSION_NAME, log_level_name)

        # 引数指定がない場合、通常起動
        app = wx.App(False)
        icon = wx.Icon(MFileUtils.resource_path("src/vroid2pmx.ico"), wx.BITMAP_TYPE_ICO)
        frame = MainFrame(None, mydir_path, now_version_name, args.verbose, is_saving, is_out_log)
        frame.SetIcon(icon)
        frame.Show(True)
        app.MainLoop()
