import os
import subprocess

try:
    version = subprocess.check_output(['git', '-C', os.path.dirname(__file__), 'describe', '--tags', '--always', '--dirty'], stderr=subprocess.PIPE).decode('ascii').strip()
except:
    from fabmos._version_setup import version as version

