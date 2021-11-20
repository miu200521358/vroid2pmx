# -*- coding: utf-8 -*-
#
from datetime import datetime
import logging
import traceback
import threading
import sys
import os
import json
import locale

import cython

from utils.MException import MKilledException


class MLogger():

    DECORATION_IN_BOX = "in_box"
    DECORATION_BOX = "box"
    DECORATION_LINE = "line"
    DEFAULT_FORMAT = "%(message)s [%(funcName)s][P-%(process)s](%(asctime)s)"

    DEBUG_FULL = 2
    TEST = 5
    TIMER = 12
    FULL = 15
    DEBUG_INFO = 16
    INFO_DEBUG = 22
    DEBUG = logging.DEBUG       # 10
    INFO = logging.INFO         # 20
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # 翻訳モード
    # 読み取り専用：翻訳リストにない文字列は入力文字列をそのまま出力する
    MODE_READONLY = 0
    # 更新あり：翻訳リストにない文字列は出力する
    MODE_UPDATE = 1

    total_level = logging.INFO
    is_file = False
    mode = MODE_READONLY
    outout_datetime = ""

    # 翻訳モード
    mode = MODE_READONLY
    # 翻訳言語優先順位
    langs = ["en_US", "ja_JP", "zh_CN"]
    # 出力対象言語
    target_lang = "ja_JP"
    
    messages = {}
    logger = None

    def __init__(self, module_name, level=logging.INFO):
        self.module_name = module_name
        self.default_level = level
        self.child = False

        # ロガー
        self.logger = logging.getLogger("VmdSizing").getChild(self.module_name)

        # 標準出力ハンドラ
        sh = logging.StreamHandler()
        sh.setLevel(level)
        # sh.setFormatter(logging.Formatter(self.DEFAULT_FORMAT))
        # sh.setStream(sys.stdout)
        self.logger.addHandler(sh)

    def copy(self, options):
        self.is_file = options.is_file
        self.outout_datetime = options.outout_datetime
        self.monitor = options.monitor
        self.child = True

        for f in self.logger.handlers:
            if isinstance(f, logging.StreamHandler):
                f.setStream(options.monitor)

    def time(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.TIMER
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)

    def info_debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.INFO_DEBUG
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)

    def debug_info(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.DEBUG_INFO
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)

    def test(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = self.TEST
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.DEBUG
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.INFO
        self.print_logger(msg, *args, **kwargs)

    # ログレベルカウント
    def count(self, msg, fno, fnos, *args, **kwargs):
        last_fno = 0

        if fnos and len(fnos) > 0 and fnos[-1] > 0:
            last_fno = fnos[-1]
        
        if not fnos and kwargs and "last_fno" in kwargs and kwargs["last_fno"] > 0:
            last_fno = kwargs["last_fno"]

        if last_fno > 0:
            if not kwargs:
                kwargs = {}
                
            kwargs["level"] = logging.INFO
            log_msg = "-- {0}フレーム目:終了({1}％){2}".format(fno, round((fno / last_fno) * 100, 3), msg)
            self.print_logger(log_msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.WARNING
        self.print_logger(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.ERROR
        self.print_logger(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.CRITICAL
        self.print_logger(msg, *args, **kwargs)

    # 実際に出力する実態
    def print_logger(self, org_msg, *args, **kwargs):

        if "is_killed" in threading.current_thread()._kwargs and threading.current_thread()._kwargs["is_killed"]:
            # 停止命令が出ている場合、エラー
            raise MKilledException()

        target_level = kwargs.pop("level", logging.INFO)
        # if self.logger.isEnabledFor(target_level) and self.default_level <= target_level:
        if self.total_level <= target_level and self.default_level <= target_level:

            if self.is_file:
                for f in self.logger.handlers:
                    if isinstance(f, logging.FileHandler):
                        # 既存のファイルハンドラはすべて削除
                        self.logger.removeHandler(f)

                # ファイル出力ありの場合、ハンドラ紐付け
                # ファイル出力ハンドラ
                fh = logging.FileHandler("log/VmdSizing_{0}.log".format(self.outout_datetime))
                fh.setLevel(self.default_level)
                fh.setFormatter(logging.Formatter(self.DEFAULT_FORMAT))
                self.logger.addHandler(fh)

            # モジュール名を出力するよう追加
            extra_args = {}
            extra_args["module_name"] = self.module_name

            # 翻訳有無で出力メッセージ取得
            is_translate = kwargs.pop("translate", True)
            msg = self.transtext(org_msg) if is_translate and target_level >= self.INFO else org_msg

            # ログレコード生成
            if args and isinstance(args[0], Exception) or (args and len(args) > 1 and isinstance(args[0], Exception)):
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, "{0}\n\n{1}".format(msg, traceback.format_exc()), None, None, self.module_name)
            else:
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, msg, args, None, self.module_name)
            
            target_decoration = kwargs.pop("decoration", None)
            title = kwargs.pop("title", None)
            is_time = kwargs.pop("time", None)

            if is_time:
                # 時間表記が必要な場合、表記追加
                print_msg = "{message} [{funcName}]({now:%H:%M:%S.%f})".format(message=log_record.getMessage(), funcName=self.module_name, now=datetime.now())
            else:
                print_msg = "{message}".format(message=log_record.getMessage())
            
            if target_decoration:
                if target_decoration == MLogger.DECORATION_BOX:
                    output_msg = self.create_box_message(print_msg, target_level, title)
                elif target_decoration == MLogger.DECORATION_LINE:
                    output_msg = self.create_line_message(print_msg, target_level, title)
                elif target_decoration == MLogger.DECORATION_IN_BOX:
                    output_msg = self.create_in_box_message(print_msg, target_level, title)
                else:
                    output_msg = self.create_simple_message(print_msg, target_level, title)
            else:
                output_msg = self.create_simple_message(print_msg, target_level, title)
        
            # 出力
            try:
                if self.child or self.is_file:
                    # 子スレッドの場合はレコードを再生成してでコンソールとGUI両方出力
                    log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, output_msg, None, None, self.module_name)
                    self.logger.handle(log_record)
                else:
                    # サイジングスレッドは、printとloggerで分けて出力
                    print_message(output_msg, target_level)
                    self.logger.handle(log_record)
            except Exception as e:
                raise e
            
    def create_box_message(self, msg, level, title=None):
        msg_block = []
        msg_block.append("■■■■■■■■■■■■■■■■■")

        if level == logging.CRITICAL:
            msg_block.append("■　**CRITICAL**　")

        if level == logging.ERROR:
            msg_block.append("■　**ERROR**　")

        if level == logging.WARNING:
            msg_block.append("■　**WARNING**　")

        if level <= logging.INFO and title:
            msg_block.append("■　**{0}**　".format(title))

        for msg_line in msg.split("\n"):
            msg_block.append("■　{0}".format(msg_line))

        msg_block.append("■■■■■■■■■■■■■■■■■")

        return "\n".join(msg_block)

    def create_line_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            msg_block.append("■■ {0} --------------------".format(msg_line))

        return "\n".join(msg_block)

    def create_in_box_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            msg_block.append("■　{0}".format(msg_line))

        return "\n".join(msg_block)

    def create_simple_message(self, msg, level, title=None):
        msg_block = []
        
        for msg_line in msg.split("\n"):
            # msg_block.append("[{0}] {1}".format(logging.getLevelName(level)[0], msg_line))
            msg_block.append(msg_line)
        
        return "\n".join(msg_block)

    def transtext(self, msg):
        trans_msg = msg
        if msg in self.messages:
            # メッセージがある場合、それで出力する
            trans_msg = self.messages[msg]

        if self.mode == MLogger.MODE_UPDATE:
            # 更新モードである場合、辞書に追記
            for lang in self.langs:
                messages_path = self.get_message_path(lang)
                try:
                    with open(messages_path, 'r', encoding="utf-8") as f:
                        msgs = json.load(f)

                        if msg not in msgs:
                            # ない場合、追加(オリジナル言語の場合、そのまま。違う場合は空欄)
                            msgs[msg] = msg if self.target_lang == lang else ""

                        with open(messages_path, 'w', encoding="utf-8") as f:
                            json.dump(msgs, f, ensure_ascii=False)
                except Exception:
                    print("*** Message Update ERROR ***\n%s", traceback.format_exc())
        
        return trans_msg

    @classmethod
    def initialize(cls, level=logging.INFO, is_file=False, mode=MODE_READONLY):
        # logging.basicConfig(level=level)
        logging.basicConfig(level=level, format=cls.DEFAULT_FORMAT)
        cls.total_level = level
        cls.is_file = is_file
        cls.mode = mode
        cls.outout_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.now())

        if mode == MLogger.MODE_UPDATE:
            # 更新版の場合、必要なディレクトリ・ファイルを全部作成する
            for lang in cls.langs:
                messages_path = cls.get_message_path(lang)
                os.makedirs(os.path.dirname(messages_path), exist_ok=True)
                if not os.path.exists(messages_path):
                    try:
                        with open(messages_path, 'w', encoding="utf-8") as f:
                            json.dump({}, f, ensure_ascii=False)
                    except Exception:
                        print("*** Message Dump ERROR ***\n%s", traceback.format_exc())

        # 実行環境に応じたローカル言語
        lang = locale.getdefaultlocale()[0]
        if lang not in cls.langs:
            # 実行環境言語に対応した言語が出力対象外である場合、第一言語を出力する
            cls.target_lang = cls.langs[0]
        else:
            # 実行環境言語に対応した言語が出力対象である場合、その言語を出力する
            # cls.target_lang = "ja_JP"
            cls.target_lang = lang

        # # 言語固定
        # cls.target_lang = ["en_US", "ja_JP", "zh_CN"][0]

        # メッセージファイルパス
        try:
            with open(cls.get_message_path(cls.target_lang), 'r', encoding="utf-8") as f:
                cls.messages = json.load(f)
        except Exception:
            print("*** Message Load ERROR ***\n%s", traceback.format_exc())

    @classmethod
    def get_message_path(cls, lang):
        return resource_path(os.path.join("src", "locale", lang, "messages.json"))


# リソースファイルのパス
def resource_path(relative):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


@cython.ccall
def print_message(msg: str, target_level: int):
    sys.stdout.write(msg + "\n", (target_level < MLogger.INFO))
