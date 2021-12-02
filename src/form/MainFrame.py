# -*- coding: utf-8 -*-
#
import os
import sys
import wx
import threading

from form.panel.FilePanel import FilePanel
from utils import MFormUtils, MFileUtils # noqa
from utils.MLogger import MLogger # noqa
from form.worker.ExportWorkerThread import ExportWorkerThread
from form.worker.LoadWorkerThread import LoadWorkerThread

if os.name == "nt":
    import winsound     # Windows版のみインポート

logger = MLogger(__name__)


# イベント
(TailorThreadEvent, EVT_TAILOR_THREAD) = wx.lib.newevent.NewEvent()
(LoadThreadEvent, EVT_LOAD_THREAD) = wx.lib.newevent.NewEvent()


class MainFrame(wx.Frame):

    def __init__(self, parent, mydir_path: str, version_name: str, logging_level: int, is_saving: bool, is_out_log: bool):
        self.version_name = version_name
        self.logging_level = logging_level
        self.is_out_log = is_out_log
        self.is_saving = is_saving
        self.mydir_path = mydir_path
        self.elapsed_time = 0
        self.popuped_finger_warning = False
        
        self.worker = None
        self.load_worker = None

        self.my_program = 'Vroid2Pmx'

        frame_size = wx.Size(600, 650)
        if logger.target_lang == "en_US":
            frame_size = wx.Size(800, 650)
        elif logger.target_lang == "zh_CN":
            frame_size = wx.Size(700, 650)

        frame_title = logger.transtext(f'{self.my_program} ローカル版') + f' {self.version_name}'
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=frame_title, \
                          pos=wx.DefaultPosition, size=frame_size, style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        # ファイル履歴読み込み
        self.file_hitories = MFileUtils.read_history(self.mydir_path)

        # ---------------------------------------------

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.VERTICAL)

        self.note_ctrl = wx.Notebook(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0)
        if self.logging_level == MLogger.FULL or self.logging_level == MLogger.DEBUG_FULL:
            # フルデータの場合
            self.note_ctrl.SetBackgroundColour("RED")
        elif self.logging_level == MLogger.DEBUG:
            # テスト（デバッグ版）の場合
            self.note_ctrl.SetBackgroundColour("CORAL")
        elif self.logging_level == MLogger.TIMER:
            # 時間計測の場合
            self.note_ctrl.SetBackgroundColour("YELLOW")
        elif not is_saving:
            # ログありの場合、色変え
            self.note_ctrl.SetBackgroundColour("BLUE")
        elif is_out_log:
            # ログありの場合、色変え
            self.note_ctrl.SetBackgroundColour("AQUAMARINE")
        else:
            self.note_ctrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW))

        # ---------------------------------------------

        # ファイルタブ
        self.file_panel_ctrl = FilePanel(self, self.note_ctrl, 0)
        self.note_ctrl.AddPage(self.file_panel_ctrl, logger.transtext("ファイル"), False)

        # ---------------------------------------------

        # タブ押下時の処理
        self.note_ctrl.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_tab_change)

        # イベントバインド
        self.Bind(EVT_TAILOR_THREAD, self.on_exec_result)
        self.Bind(EVT_LOAD_THREAD, self.on_load_result)

        # ---------------------------------------------

        bSizer1.Add(self.note_ctrl, 1, wx.EXPAND, 5)

        # デフォルトの出力先はファイルタブのコンソール
        sys.stdout = self.file_panel_ctrl.console_ctrl

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)
    
    def on_idle(self, event: wx.Event):
        pass

    def on_tab_change(self, event: wx.Event):

        if self.file_panel_ctrl.is_fix_tab:
            self.note_ctrl.ChangeSelection(self.file_panel_ctrl.tab_idx)
            event.Skip()
            return

    # タブ移動可
    def release_tab(self):
        self.file_panel_ctrl.release_tab()

    # フォーム入力可
    def enable(self):
        self.file_panel_ctrl.enable()

    def show_worked_time(self):
        # 経過秒数を時分秒に変換
        td_m, td_s = divmod(self.elapsed_time, 60)

        if td_m == 0:
            if logger.target_lang == "ja_JP":
                worked_time = "{0:02d}秒".format(int(td_s))
            else:
                worked_time = "{0:02d}s".format(int(td_s))
        else:
            if logger.target_lang == "ja_JP":
                worked_time = "{0:02d}分{1:02d}秒".format(int(td_m), int(td_s))
            else:
                worked_time = "{0:02d}m{1:02d}s".format(int(td_m), int(td_s))

        return worked_time
    
    # ファイルセットの入力可否チェック
    def is_valid(self):
        result = True
        result = self.file_panel_ctrl.org_model_file_ctrl.is_valid() and result

        return result
    
    # 読み込み
    def load(self, event, is_exec=False, is_param=False, is_param_advance=False, is_param_bone=False):
        # フォーム無効化
        self.file_panel_ctrl.disable()
        # タブ固定
        self.file_panel_ctrl.fix_tab()

        self.elapsed_time = 0
        result = True
        result = self.is_valid() and result

        if not result:
            if is_param or is_param_advance or is_param_bone:
                tab_name = logger.transtext("パラ調整")
                if is_param_advance:
                    tab_name = logger.transtext("パラ調整(詳細)")
                if is_param_bone:
                    tab_name = logger.transtext("パラ調整(ボーン)")
                # 読み込み出来なかったらエラー
                logger.error("「ファイル」タブで対象モデルファイルパスが指定されていないため、「%s」タブが開けません。" \
                             + "\n既に指定済みの場合、現在読み込み中の可能性があります。" \
                             + "\n「■読み込み成功」のログが出てから、「%s」タブを開いてください。", tab_name, tab_name, decoration=MLogger.DECORATION_BOX)

            # タブ移動可
            self.release_tab()
            # フォーム有効化
            self.enable()

            return result

        # 読み込み開始
        if self.load_worker:
            logger.error(logger.transtext("まだ処理が実行中です。終了してから再度実行してください。"), decoration=MLogger.DECORATION_BOX)
        else:
            # 停止ボタンに切り替え
            self.file_panel_ctrl.export_btn_ctrl.SetLabel(logger.transtext("読み込み処理停止"))
            self.file_panel_ctrl.export_btn_ctrl.Enable()

            # 別スレッドで実行
            self.load_worker = LoadWorkerThread(self, LoadThreadEvent, is_exec, is_param, is_param_advance, is_param_bone)
            self.load_worker.start()

        return result

    def is_loaded_valid(self):
        return True

    # 読み込み完了処理
    def on_load_result(self, event: wx.Event):
        self.elapsed_time = event.elapsed_time
        
        # タブ移動可
        self.release_tab()
        # フォーム有効化
        self.enable()
        # ワーカー終了
        self.load_worker = None
        # プログレス非表示
        self.file_panel_ctrl.gauge_ctrl.SetValue(0)

        # チェックボタンに切り替え
        self.file_panel_ctrl.export_btn_ctrl.SetLabel(self.file_panel_ctrl.txt_exec)
        self.file_panel_ctrl.export_btn_ctrl.Enable()

        if not event.result:
            # 終了音を鳴らす
            self.sound_finish()
            # タブ移動可
            self.release_tab()
            # フォーム有効化
            self.enable()

            event.Skip()
            return False
        
        logger.info(logger.transtext("ファイルデータ読み込みが完了しました"), decoration=MLogger.DECORATION_BOX, title="OK")

        if event.is_exec:
            if not self.is_loaded_valid():
                # 終了音を鳴らす
                self.sound_finish()
                # タブ移動可
                self.release_tab()
                # フォーム有効化
                self.enable()

                event.Skip()
                return False

            # そのまま実行する場合、変換実行処理に遷移

            # フォーム無効化
            self.file_panel_ctrl.disable()
            # タブ固定
            self.file_panel_ctrl.fix_tab()

            if self.worker:
                logger.error(logger.transtext("まだ処理が実行中です。終了してから再度実行してください。"), decoration=MLogger.DECORATION_BOX)
            else:
                # 停止ボタンに切り替え
                self.file_panel_ctrl.export_btn_ctrl.SetLabel(self.file_panel_ctrl.txt_stop)
                self.file_panel_ctrl.export_btn_ctrl.Enable()

                # 別スレッドで実行
                self.worker = ExportWorkerThread(self, TailorThreadEvent, self.is_saving, self.is_out_log)
                self.worker.start()

        elif event.is_param:
            # パラ調整タブを開く場合、パラ調整タブ初期化処理実行
            self.note_ctrl.ChangeSelection(self.simple_param_panel_ctrl.tab_idx)
            self.simple_param_panel_ctrl.initialize(event)

        elif event.is_param_advance:
            # パラ調整(詳細)タブを開く場合、パラ調整(詳細)タブ初期化処理実行
            self.note_ctrl.ChangeSelection(self.advance_param_panel_ctrl.tab_idx)
            self.advance_param_panel_ctrl.initialize(event)

        elif event.is_param_bone:
            # パラ調整(ボーン)タブを開く場合、パラ調整(ボーン)タブ初期化処理実行
            self.note_ctrl.ChangeSelection(self.bone_param_panel_ctrl.tab_idx)
            self.bone_param_panel_ctrl.initialize(event)

        else:
            # 終了音を鳴らす
            self.sound_finish()

            logger.info("\n処理時間: %s", self.show_worked_time())
        
            event.Skip()
            return True

    # スレッド実行結果
    def on_exec_result(self, event: wx.Event):
        # 実行ボタンに切り替え
        self.file_panel_ctrl.export_btn_ctrl.SetLabel(self.file_panel_ctrl.txt_exec)
        self.file_panel_ctrl.export_btn_ctrl.Enable()

        self.elapsed_time += event.elapsed_time

        logger.info("\n処理時間: %s", self.show_worked_time())

        # ワーカー終了
        self.worker = None

        # ファイルタブのコンソール
        sys.stdout = self.file_panel_ctrl.console_ctrl

        # 終了音を鳴らす
        self.sound_finish()

        # タブ移動可
        self.release_tab()
        # フォーム有効化
        self.enable()
        # プログレス非表示
        self.file_panel_ctrl.gauge_ctrl.SetValue(0)

    def sound_finish(self):
        threading.Thread(target=self.sound_finish_thread).start()

    def sound_finish_thread(self):
        # 終了音を鳴らす
        if os.name == "nt":
            # Windows
            try:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
            except Exception:
                pass

    def on_wheel_spin_ctrl(self, event: wx.Event, inc=0.1):
        # スピンコントロール変更時
        if event.GetWheelRotation() > 0:
            event.GetEventObject().SetValue(event.GetEventObject().GetValue() + inc)
            if event.GetEventObject().GetValue() >= 0:
                event.GetEventObject().SetBackgroundColour("WHITE")
        else:
            event.GetEventObject().SetValue(event.GetEventObject().GetValue() - inc)
            if event.GetEventObject().GetValue() < 0:
                event.GetEventObject().SetBackgroundColour("TURQUOISE")

        # スピンコントロールでもタイムスタンプ変更
        self.file_panel_ctrl.on_change_file(event)
