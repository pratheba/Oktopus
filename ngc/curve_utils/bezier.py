import numpy as np


def bezier_curve(p0, p1, p2, p3, t):
    t = np.asarray(t)
    p = 1.0 - t
    b_curve = (p**3)[:, None]*p0 +\
            (3*p**2*t)[:, None]*p1 + \
            (3*p*t**2)[:, None]*p2 + \
            (t**3)[:, None]*p3
    return b_curve


def sample_bezier_uniform(p0, p1, p2, p3, n_key=100, n_dense=5000):
    pass



if __name__ == '__main__':
    okto_tentacle = 
