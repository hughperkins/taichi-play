import taichi as ti
import time
import math
import numpy as np
# n = 320

# N = n * 2 * n
N = int(1e6)

# pixels = np.zeros((n * 2, n))

@ti.func
def is_prime(i: int) -> bool:
	# if i in [0, 1, 2, 3]:
	known_prime = False
	known_not_prime = False
	for j in ti.static([0, 1, 2, 3]):
		if i == j:
			known_prime = True
	if i > 2 and i % 2 == 0:
		known_not_prime = True
	j = 3
	max_j = int(ti.sqrt(i)) + 1
	while j <= max_j:
		if i % j == 0:
			known_not_prime = True
		j += 2
	return known_prime or not known_not_prime

@ti.kernel
def count_primes(N: int) -> int:
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
	return count

ti.init(arch=ti.gpu)
start = time.time()
count = count_primes(N)
elapsed = time.time() - start
# print(pixels[:20, :10])
print('count', count)
print('elapsed', elapsed)
