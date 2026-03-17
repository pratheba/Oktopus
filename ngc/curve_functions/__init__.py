import os, sys
#current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
#sys.path.append(current_path)

from ._interpolate import *

