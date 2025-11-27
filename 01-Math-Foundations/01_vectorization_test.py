import numpy as np
import time


print("Day 1: The Speed of Vectorization")

def dot_product_loops(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def dot_product_numpy(a, b):
    return np.dot(a, b)

def main():
    size = 1_000_000
    array_1 = np.random.rand(size)
    array_2 = np.random.rand(size)

    start_loop = time.time()
    result_loop = dot_product_loops(array_1, array_2)
    end_loop = time.time()

    print(f"Loop Result: {result_loop:.4f}")
    print(f"Loop Time:   {end_loop - start_loop:.6f} seconds")

    start_numpy = time.time()
    result_numpy = dot_product_numpy(array_1, array_2)
    end_numpy = time.time()

    print(f"NumPy Result: {result_numpy:.4f}")
    print(f"NumPy Time:   {end_numpy - start_numpy:.6f} seconds")

    speedup = (end_loop - start_loop) / (end_numpy - start_numpy)
    print(f"Vectorization Speedup: {speedup:.2f}x faster")

if __name__ == "__main__":
    main()