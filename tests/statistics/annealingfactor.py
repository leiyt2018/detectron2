import math
import numpy as np

t = np.arange(1,51).astype(np.float)
lambda0 = 0.01
T = 50.0
# tmp = np.log((lambda0/T)*t)
lambdat = lambda0 * np.exp((-1)*np.log(lambda0/T)*t)

print(lambdat)