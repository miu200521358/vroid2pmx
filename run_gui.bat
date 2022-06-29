@echo off
rem --- 
rem ---  vmdデータのトレースモデルを変換
rem --- 

rem ---  カレントディレクトリを実行先に変更
cd /d %~dp0

cls

set LANG=%~1
if "%LANG%"=="" (set LANG=ja_JP)

activate vmdsizing_cython && src\setup.bat && python src\executor.py --out_log 1 --verbose 20 --is_saving 1 --lang %LANG%

