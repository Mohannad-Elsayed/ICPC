#!/usr/bin/env python3

# Absolute worst-case style input for the submitted O(n^2)-ish solution.
#
# The only valid n are 2^h - 1 because:
#   - every subtree of a full binary tree has odd size
#   - both child subtree sizes are odd
#   - their difference <= 1 => they are equal
#
# The sizes below maximize the actual number of inner-loop iterations
# of mul() subject to sum(n) <= 100000.

SIZES = [
    65535,
    32767,
    1023,
    511,
    127,
    31,
    3,
    3,
]

assert sum(SIZES) == 100000
assert all((n & (n + 1)) == 0 for n in SIZES)

print(len(SIZES))

for n in SIZES:
    print(n)

    for u in range(1, (n - 1) // 2 + 1):
        print(u, 2 * u)
        print(u, 2 * u + 1)