import random

def generate_tle_stress_test(filename="5"):
    # Maximum allowed odd length under standard 2e5 limits
    n = 199999

    # A random permutation of distinct elements forces widespread
    # scattering of X[i] and Y[i] values.
    # This guarantees massive cross-boundary DP updates in the CDQ tree,
    # strictly requiring O(1) rollbacks per operation to pass.
    a = list(range(1, n + 1))

    # Fixed seed for reproducibility (optional)
    random.seed(42)
    random.shuffle(a)

    with open(filename, 'w') as f:
        f.write("1\n") # 1 test case
        f.write(f"{n}\n")
        f.write(" ".join(map(str, a)) + "\n")

    print(f"Generated Fenwick stress test data in '{filename}' (N={n})")

if __name__ == "__main__":
    generate_tle_stress_test()