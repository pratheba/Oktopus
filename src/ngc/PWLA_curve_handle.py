import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator
from handle_utils import CylindersMesh
from scipy.ndimage import gaussian_filter1d
from curve_utils.visualize_util import *
from curve_utils.curve_utils import *
from curve_functions._interpolate import interpolate_occ_profile1, interpolate_wrap_radius1
from curve_functions._frame import *
from curve_functions._update import update_wrap_profile_from_coords, update_wrap_occupancy_from_coords
#from curve_functions._localize import localize_samples


#n_sample_curve = 200
#n_sample_circle = 120

n_sample_curve = 200
n_sample_circle = 120
#n_sample_points = 12


from pwla_curve import (_CoreMixin, _FramesMixin, _RadiusMixin, _WrapMixin, _GeometryMixin, _LocalizeMixin, _StretchMixin, _InterpolateMixin)

class PWLACurve(_CoreMixin, _FramesMixin, _RadiusMixin, _WrapMixin, _GeometryMixin, _StretchMixin, _LocalizeMixin, _InterpolateMixin):
    """docstring for PWLACurve."""
    pass
