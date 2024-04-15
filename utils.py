import numpy as np

def l2_distance(real,fake):
    return np.linalg.norm(real-fake)


def get_infinite_batch(dataloader):
    while True:
        for data in dataloader:
            yield data