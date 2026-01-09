# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

block_cipher = None

# Collect textual CSS
# In source: src/bat/tui/styles.tcss
# We map it to bat/tui/styles.tcss in the bundle so relative imports work
datas = [
    ('src/bat/tui/styles.tcss', 'bat/tui'),
]

# Copy metadata for key packages
datas += copy_metadata('textual')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('pandas')
datas += copy_metadata('torch')

hiddenimports = [
    'bat', 
    'bat.tui', 
    'bat.tui.app', 
    'bat.execution', 
    'bat.models', 
    'bat.data', 
    'pandas', 
    'numpy', 
    'sklearn', 
    'torch',
    'pandas_ta',
    'textual.widgets',
    'textual.containers',
]

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'PyQt5', 'pyside6', 'PySide6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BAT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
