import numpy as np

def l2_distance(real,fake):
    return np.linalg.norm(real-fake)