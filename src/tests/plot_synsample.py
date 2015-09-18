import numpy as np
import matplotlib.pyplot as plt

from .test_pipeline import SynSample2
from ..plot import histnd

J = 4
n_obs = 1000
d = 3
ver = 'B'
samples = [SynSample2(j, n_obs, d=3, ver=ver) for j in range(J)]

histnd(np.vstack([sample.data for sample in samples]), 50)
plt.show()
