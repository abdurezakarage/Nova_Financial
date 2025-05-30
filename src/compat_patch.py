from packaging.version import Version as LooseVersion
import sys
import os

# Add this directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Monkey patch pandas_datareader's compat module
import pandas_datareader.compat
pandas_datareader.compat.LooseVersion = LooseVersion