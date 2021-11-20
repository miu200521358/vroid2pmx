# -*- coding: utf-8 -*-
#

from datetime import datetime
import sys
import os
import json
import glob
from pathlib import Path
import re
import _pickle as cPickle
import shutil

from utils.MLogger import MLogger # noqa

logger = MLogger(__name__)


# リソースファイルのパス
def resource_path(relative):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


# ファイル履歴読み込み
def read_history(mydir_path):
    # ファイル履歴
    base_file_hitories = {"org_pmx": [], "org_vroid": [], "vertices_csv": [], "max": 50}
    file_hitories = cPickle.loads(cPickle.dumps(base_file_hitories, -1))

    # 履歴JSONファイルがあれば読み込み
    try:
        with open(os.path.join(mydir_path, 'history.json'), 'r', encoding="utf-8") as f:
            file_hitories = json.load(f)
            # キーが揃っているかチェック
            for key in base_file_hitories.keys():
                if key not in file_hitories:
                    file_hitories[key] = []
            # 最大件数は常に上書き
            file_hitories["max"] = 50
    except Exception:
        # UTF-8で読み込めなかった場合、デフォルトで読み込んでUTF-8変換
        try:
            with open(os.path.join(mydir_path, 'history.json'), 'r') as f:
                file_hitories = json.load(f)
                # キーが揃っているかチェック
                for key in base_file_hitories.keys():
                    if key not in file_hitories:
                        file_hitories[key] = []
                # 最大件数は常に上書き
                file_hitories["max"] = 50
            
            # 一旦UTF-8で出力
            save_history(mydir_path, file_hitories)

            # UTF-8で読み込みし直し
            return read_history(mydir_path)
        except Exception:
            file_hitories = cPickle.loads(cPickle.dumps(base_file_hitories, -1))

    return file_hitories


def save_history(mydir_path, file_hitories):
    # 入力履歴を保存
    try:
        with open(os.path.join(mydir_path, 'history.json'), 'w', encoding="utf-8") as f:
            json.dump(file_hitories, f, ensure_ascii=False)
    except Exception as e:
        logger.error("履歴ファイルの保存に失敗しました", e, decoration=MLogger.DECORATION_BOX)


# パス解決
def get_mydir_path(exec_path):
    logger.test("sys.argv %s", sys.argv)
    
    dir_path = Path(exec_path).parent if hasattr(sys, "frozen") else Path(__file__).parent
    logger.test("get_mydir_path: %s", get_mydir_path)

    return dir_path


# ディレクトリパス
def get_dir_path(base_file_path, is_print=True):
    if os.path.exists(base_file_path):
        file_path_list = [base_file_path]
    else:
        file_path_list = [p for p in glob.glob(base_file_path) if os.path.isfile(p)]

    if len(file_path_list) == 0:
        return ""

    try:
        # ファイルパスをオブジェクトとして解決し、親を取得する
        return str(Path(file_path_list[0]).resolve().parents[0])
    except Exception as e:
        logger.error("ファイルパスの解析に失敗しました。\nパスに使えない文字がないか確認してください。\nファイルパス: {0}\n\n{1}".format(base_file_path, e.with_traceback(sys.exc_info()[2])))
        raise e
    

# PMX出力ファイルパス生成
# org_pmx_path: 変換先モデルVRMパス
# output_pmx_path: 出力PMXファイルパス
def get_output_pmx_path(org_pmx_path: str, output_pmx_path: str, is_force=False):
    if not os.path.exists(org_pmx_path):
        return ""

    # VRMディレクトリパス
    pmx_dir_path = get_dir_path(org_pmx_path)
    # VRMモデルファイル名・拡張子
    org_pmx_file_name, _ = os.path.splitext(os.path.basename(org_pmx_path))

    # 出力ファイルパス生成
    new_output_pmx_path = os.path.join(pmx_dir_path, f'{org_pmx_file_name}_{datetime.now():%Y%m%d_%H%M%S}.pmx')

    # ファイルパス自体が変更されたか、自動生成ルールに則っている場合、ファイルパス変更
    if is_force or re.match(r'%s_\d{8}_\d{6}.pmx' % (org_pmx_file_name), output_pmx_path) is not None:

        try:
            open(new_output_pmx_path, 'w')
            os.remove(new_output_pmx_path)
        except Exception:
            logger.warning("出力ファイルパスの生成に失敗しました。以下の原因が考えられます。\n" \
                           + "・ファイルパスが255文字を超えている\n" \
                           + "・ファイルパスに使えない文字列が含まれている（例) \\　/　:　*　?　\"　<　>　|）" \
                           + "・出力ファイルパスの親フォルダに書き込み権限がない" \
                           + "・出力ファイルパスに書き込み権限がない")

        return new_output_pmx_path

    return output_pmx_path
    

# Vroid2Pmx出力ファイルパス生成
# org_pmx_path: 変換先モデルVRMパス
# output_pmx_path: 出力PMXファイルパス
def get_output_vrm_path(org_pmx_path: str, output_pmx_path: str, is_force=False):
    if not os.path.exists(org_pmx_path):
        return ""

    # VRMディレクトリパス
    pmx_dir_path = get_dir_path(org_pmx_path)
    # VRMモデルファイル名・拡張子
    org_pmx_file_name, _ = os.path.splitext(os.path.basename(org_pmx_path))

    # 出力ファイルパス生成
    new_output_pmx_path = os.path.join(pmx_dir_path, org_pmx_file_name, f'{datetime.now():%Y%m%d_%H%M%S}', f'{org_pmx_file_name}.pmx')

    # ファイルパス自体が変更されたか、自動生成ルールに則っている場合、ファイルパス変更
    if is_force or re.match(r'\d{8}_\d{6}\\%s\.pmx' % (org_pmx_file_name), output_pmx_path) is not None:

        try:
            os.makedirs(os.path.dirname(new_output_pmx_path), exist_ok=True)
            open(new_output_pmx_path, 'w')
            shutil.rmtree(os.path.dirname(new_output_pmx_path))
        except Exception:
            logger.warning("出力ファイルパスの生成に失敗しました。以下の原因が考えられます。\n" \
                           + "・ファイルパスが255文字を超えている\n" \
                           + "・ファイルパスに使えない文字列が含まれている（例) \\　/　:　*　?　\"　<　>　|）" \
                           + "・出力ファイルパスの親フォルダに書き込み権限がない" \
                           + "・出力ファイルパスに書き込み権限がない")

        return new_output_pmx_path

    return output_pmx_path


def escape_filepath(path: str):
    path = path.replace("\\", "\\\\")
    path = path.replace("*", "\\*")
    path = path.replace("+", "\\+")
    path = path.replace(".", "\\.")
    path = path.replace("?", "\\?")
    path = path.replace("{", "\\{")
    path = path.replace("}", "\\}")
    path = path.replace("(", "\\(")
    path = path.replace(")", "\\)")
    path = path.replace("[", "\\[")
    path = path.replace("]", "\\]")
    path = path.replace("{", "\\{")
    path = path.replace("^", "\\^")
    path = path.replace("$", "\\$")
    path = path.replace("-", "\\-")
    path = path.replace("|", "\\|")
    path = path.replace("/", "\\/")

    return path
