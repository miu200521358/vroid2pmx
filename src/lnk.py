import sys
import os
import threading
import subprocess
from executor import VERSION_NAME

APP_NAME = "Vroid2Pmx"

if __name__ == "__main__":
    for file_path, cmd, icon_loc in [
        (
            f"..\dist\{APP_NAME}_{VERSION_NAME}.exe - en_US.lnk",
            f'/c start "%cd%" "{APP_NAME}_{VERSION_NAME}.exe" --verbose 20 --out_log 0 --lang en_US',
            f'"%SystemRoot%\System32\SHELL32.dll, 0"',
        ),
        (
            f"..\dist\{APP_NAME}_{VERSION_NAME}.exe - en_US_Log.lnk",
            f'/c start "%cd%" "{APP_NAME}_{VERSION_NAME}.exe" --verbose 20 --out_log 1 --lang en_US',
            f'"%SystemRoot%\System32\SHELL32.dll, 0"',
        ),
        (
            f"..\dist\{APP_NAME}_{VERSION_NAME}.exe - zh_CN.lnk",
            f'/c start "%cd%" "{APP_NAME}_{VERSION_NAME}.exe" --verbose 20 --out_log 0 --lang zh_CN',
            f'"%SystemRoot%\System32\SHELL32.dll, 0"',
        ),
        (
            f"..\dist\{APP_NAME}_{VERSION_NAME}.exe - zh_CN_Log.lnk",
            f'/c start "%cd%" "{APP_NAME}_{VERSION_NAME}.exe" --verbose 20 --out_log 1 --lang zh_CN',
            f'"%SystemRoot%\System32\SHELL32.dll, 0"',
        ),
        (
            f"..\dist\{APP_NAME}_{VERSION_NAME}.exe - デバッグ版.lnk",
            f'/c start "%cd%" "{APP_NAME}_{VERSION_NAME}.exe" --verbose 10 --out_log 1 --lang ja_JP',
            f'"%SystemRoot%\System32\SHELL32.dll, 0"',
        ),
        (
            f"..\dist\{APP_NAME}_{VERSION_NAME}.exe - ログあり版.lnk",
            f'/c start "%cd%" "{APP_NAME}_{VERSION_NAME}.exe" --verbose 20 --out_log 1 --lang ja_JP',
            f'"%SystemRoot%\System32\SHELL32.dll, 0"',
        ),
    ]:
        print("file_name: ", file_path)
        print("cmd: ", cmd)
        print("icon_loc: ", icon_loc)

        proc = subprocess.Popen(
            [
                "cscript",
                os.path.abspath("lnk.vbs"),
                os.path.abspath(file_path),
                "%windir%\system32\cmd.exe ",
                cmd,
                icon_loc,
            ],
        )
        proc.wait()
        proc.terminate()

    print("finish")
    sys.exit()
