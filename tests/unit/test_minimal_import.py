"""A base install must not require optional OCR packages just to import."""
import os
import subprocess
import sys


def test_import_without_numpy_or_easyocr():
    program = '''
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'easyocr'}:
        raise ImportError('optional OCR dependency blocked by minimal-install test')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from extract_thinker import Extractor, DocumentLoaderPyPdf
assert Extractor and DocumentLoaderPyPdf
'''
    env = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP='True')
    subprocess.run([sys.executable, '-c', program], env=env, check=True,
                   capture_output=True, text=True, timeout=60)
