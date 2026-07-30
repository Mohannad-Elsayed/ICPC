def generate_testcase():
    K = 19
    N = 2000
    edges = []

    # Construct the exponential gadget
    # A_k nodes will be mapped to IDs: k + 1 (for k=0 to K)
    # B_k nodes will be mapped to IDs: (K + 1) + k (for k=1 to K)
    for k in range(1, K + 1):
        A_k = k + 1
        A_prev = k
        B_k = K + 1 + k

        # Branch 1: Evaluated immediately
        edges.append((A_k, A_prev, 0))
        # Branch 2: Delayed positive weight
        edges.append((A_k, B_k, 1 << (k - 1)))
        # Re-entry: Heavy negative weight triggering re-evaluation
        edges.append((B_k, A_prev, -(1 << k)))

    # Add extra nodes to trigger the O(2^K) gadget from N different starting points
    A_K = K + 1
    for i in range(2 * K + 2, N + 1):
        edges.append((i, A_K, 0))

    # Print to standard output
    print(f"{N} {len(edges)}")
    for u, v, w in edges:
        print(f"{u} {v} {w}")

if __name__ == '__main__':
    print(100)
    for _ in range(100):
        generate_testcase()