import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
parent_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(parent_path)
parent_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(parent_path)


#from ngc.handle_utils import visualize


