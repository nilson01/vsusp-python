"""Fixed coefficient draws: selection, tuning and inclusion summaries."""

import numpy as np
from vsusp import s2m, two_means, spike_slab, savs, dss

beta = np.array(
    [
        [0.01, 0.02, 0.03, 0.3, 0.31, 10, 11],
        [-0.01, -0.02, -0.03, -0.3, -0.31, -10, -11],
    ]
)
path = s2m(beta, [0.1, 0.5])
print(path.summary())
print(path.select(1).selected_names)
print(two_means(beta).summary())
spike = np.array([[1, 0, 0], [2, 0, 0], [0, -1, 0], [0, 0, 0.0]])
print(spike_slab(spike).summary())
print(savs(np.eye(2), [2, 0.1]).summary())
print(dss(np.eye(2), [2, 0.1], penalty=0.2).summary())
