import numpy as np
import time
from timeit import default_timer as tic

a = np.zeros((800, 800))
rand_inds = np.random.randint(0, 800*800, 1000)
xr1, yr1 = np.unravel_index(rand_inds, a.shape)
a[xr1, yr1] = 1
rinds2 = np.random.randint(0, 800*800, 1000)
xr2, yr2 = np.unravel_index(rinds2, a.shape)
# st = time.time()
st = tic()
ans = np.sum(a[xr2, yr2])
# et = time.time()
et = tic()
print("Time for np.sum() is ", et - st)
print(ans)

# st2 = time.time()
st2 = tic()
im2 = np.zeros_like(a)
im2[xr2, yr2] = True
ans2 = np.sum(np.logical_and(a, im2))
# et2 = time.time()
et2 = tic()
print("Time for creation and logical_and is ", et2 - st2)
print(ans2)

print("logical_and() execution time is ", (et2 - st2) / (et - st), " times execution time of np.sum()")

