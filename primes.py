import time
import math
import numpy as np
# n = 320

# N = n * 2 * n
N = int(1e6)

# pixels = np.zeros((n * 2, n))

def is_prime(i):
	if i in [0, 1, 2, 3]:
		return True
	if i % 2 == 0:
		return False
	j = 3
	max_j = int(math.sqrt(i))
	while j <= max_j:
		if i % j == 0:
			return False
		j += 2
	return True

start = time.time()
count = 0
for i in range(N):
# for i in range(10):
	# print('i', i)
	if is_prime(i):
		# y = int(i / (n * 2))
		# x = i - y * (n * 2)
		count += 1
		# pixels[x, y] = 1
		# print('... prime')
elapsed = time.time() - start
# print(pixels[:20, :10])
print('count', count)
print('elapsed', elapsed)
