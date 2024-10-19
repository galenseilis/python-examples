import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import numpy as np

N = 100_000_000
REPLICATES = 1
STEP = 10_000

distances = []
sample_sizes = []
for i in range(1,N,STEP):
	for j in range(REPLICATES):
		x = np.random.normal(size=i).reshape(-1,1)
		y = np.random.normal(size=i).reshape(-1,1)
		distances.append(max(directed_hausdorff(x,y)[0], directed_hausdorff(y,x)[0]))
		sample_sizes.append(i)


plt.scatter(sample_sizes, distances, alpha=0.5)
plt.show()
