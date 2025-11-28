import numpy as np

def main():
    print("--- Day 3: Broadcasting Logic ---\n")

    # 1. Scenario & Setup
    # X represents a batch of data: 5 samples, each having 4 features.
    print("Generating Feature Matrix X (5 samples, 4 features)...")
    X = np.random.rand(5, 4)
    
    # b represents a Bias term. In ML, we often add one bias per feature.
    print("Generating Bias Vector b (4,)...")
    b = np.random.rand(4)

    # 2. Correct Broadcasting (Column-wise addition)
    print("\n--- Operation 1: Adding Bias (X + b) ---")
    # Explanation:
    # X shape: (5, 4)
    # b shape:    (4,)
    # NumPy aligns the last dimensions (4 and 4 match). 
    # It strictly 'broadcasts' b down across the 5 rows.
    Y = X + b
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of b: {b.shape}")
    print(f"Result Y shape: {Y.shape}")
    print("✅ Success: b was 'stretched' vertically to match X's 5 rows.")

    # 3. The Trap (Dimension Mismatch)
    print("\n--- Operation 2: The Trap (X + noise) ---")
    # Scenario: We want to add specific noise to each SAMPLE (row).
    # Noise shape is (5,), one value for each of the 5 samples.
    noise = np.random.rand(5)
    
    print(f"Shape of noise: {noise.shape}")
    print("Attempting: X + noise...")

    try:
        # This fails because NumPy compares dimensions right-to-left.
        # X:      (5, 4)
        # noise:     (5,)  <-- 5 != 4. Mismatch!
        bad_result = X + noise
    except ValueError as e:
        print(f"❌ Error caught: {e}")
        print("   (NumPy tried to align the 5 in noise with the 4 in X.)")

    # 4. The Fix (Reshaping for Row-wise addition)
    print("\n--- Operation 3: The Fix (Reshaping) ---")
    
    # We need to make the noise a "Column Vector" so it matches the rows of X.
    # Target shape: (5, 1)
    noise_reshaped = noise.reshape(5, 1)
    
    print(f"New noise shape: {noise_reshaped.shape}")
    
    # Now the right-to-left comparison works:
    # X:               (5, 4)
    # noise_reshaped:  (5, 1)
    #                   ^  ^
    #           (5 matches 5) (1 stretches to 4)
    
    Z = X + noise_reshaped
    print(f"Result Z shape: {Z.shape}")
    print("✅ Success: noise was 'stretched' horizontally to match X's 4 columns.")

if __name__ == "__main__":
    main()