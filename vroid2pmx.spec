# -*- coding: utf-8 -*-
# -*- mode: python -*-
# PmxTailor 64bit版

block_cipher = None


a = Analysis(['src\\executor.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=['pkg_resources', 'wx._adv', 'wx._html', 'bezier', 'quaternion', 'PIL', 'module.MParams', 'utils.MBezierUtils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['mkl','libopenblas', 'tkinter', 'win32comgenpy', 'traitlets', 'IPython', 'pydoc', 'lib2to3', 'pygments', 'matplotlib'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
a.datas += [('.\\src\\vroid2pmx.ico','.\\src\\vroid2pmx.ico', 'Data'), ('.\\src\\resources\\cheek_dye.png','.\\src\\resources\\cheek_dye.png', 'Data'), ('.\\src\\resources\\eye_heart.png','.\\src\\resources\\eye_heart.png', 'Data'), ('.\\src\\resources\\eye_star.png','.\\src\\resources\\eye_star.png', 'Data'), ('.\\src\\resources\\eye_hau.png','.\\src\\resources\\eye_hau.png', 'Data'), ('.\\src\\resources\\eye_hachume.png','.\\src\\resources\\eye_hachume.png', 'Data'), ('.\\src\\resources\\eye_nagomi.png','.\\src\\resources\\eye_nagomi.png', 'Data'), ('.\\src\\locale\\en_US\\messages.json','.\\src\\locale\\en_US\\messages.json', 'Data'), ('.\\src\\locale\\ja_JP\\messages.json','.\\src\\locale\\ja_JP\\messages.json', 'Data'), ('.\\src\\locale\\zh_CN\\messages.json','.\\src\\locale\\zh_CN\\messages.json', 'Data')]
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Vroid2Pmx_2.01.01',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False,
          icon='.\\src\\vroid2pmx.ico')

