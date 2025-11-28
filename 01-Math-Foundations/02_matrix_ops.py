import numpy as np

def multiply_matrices(A, B):
    # Get dimensions
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # Pre-check (optional in pure math, but good for debugging logic)
    if cols_A != rows_B:
        raise ValueError(f"Cannot multiply: {cols_A} (cols_A) != {rows_B} (rows_B)")

    # Initialize result matrix C of size (rows_A, cols_B) with zeros
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # The 3 Nested Loops
    for i in range(rows_A):          # Iterate through rows of A
        for j in range(cols_B):      # Iterate through columns of B
            for k in range(cols_A):  # Iterate through shared dimension
                C[i][j] += A[i][k] * B[k][j]
                
    return C

def main():
    print("--- Day 2: Matrix Multiplication & Dimensions ---\n")

    # 1. Setup Data
    rows_A, cols_A = 3, 4
    rows_B, cols_B = 4, 5
    
    print(f"Generating Matrix A ({rows_A}x{cols_A}) and Matrix B ({rows_B}x{cols_B})...")
    # Using np.random to generate data, but converting to list for the manual function 
    A_np = np.random.rand(rows_A, cols_A)
    B_np = np.random.rand(rows_B, cols_B)
    
    A_list = A_np.tolist()
    B_list = B_np.tolist()

    # 2. Manual Implementation
    print("Computing manually (3 nested loops)...")
    manual_result = multiply_matrices(A_list, B_list)

    # 3. NumPy Implementation
    print("Computing using NumPy (@ operator)...")
    numpy_result = A_np @ B_np

    # 4. Verification
    print("Verifying results...")
    # Convert manual result back to numpy array for comparison
    is_identical = np.allclose(manual_result, numpy_result)
    
    if is_identical:
        print("✅ SUCCESS: Manual and NumPy results match perfectly.")
    else:
        print("❌ FAILURE: Results do not match.")
    
    print("-" * 30)

    # 5. The Crash Test
    print("Running The Crash Test...")
    
    # Create matrix A (3, 4) and B (4, 5)
    print(f"Matrix A shape: {A_np.shape}")
    print(f"Matrix B shape: {B_np.shape}")
    print("Attempting operation: B @ A  -->  (4, 5) @ (3, 4)")

    try:
        # Attempt to compute B @ A
        # Inner dimensions will be 5 and 3, which is a mismatch.
        crash_result = B_np @ A_np
        print("Wait... this should have failed!") 
    except ValueError as e:
        print(f"Error caught: Dimension mismatch [{e}]")

if __name__ == "__main__":
    main()