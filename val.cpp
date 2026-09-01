#include "testlib.h"
#include <vector>
#include <numeric>

using namespace std;

const int MAX_N = 100'000;
const int MAX_A = 100'000;

// Disjoint Set Union (DSU) to verify the graph is a single connected tree without cycles
struct DSU {
    vector<int> p;
    DSU(int n) {
        p.resize(n + 1);
        iota(p.begin(), p.end(), 0);
    }
    int get(int x) {
        return p[x] == x ? x : p[x] = get(p[x]);
    }
    bool unite(int x, int y) {
        x = get(x);
        y = get(y);
        if (x == y) return false;
        p[x] = y;
        return true;
    }
};

int main(int argc, char* argv[]) {
    registerValidation(argc, argv);

    // 1. Read the number of vertices (n)
    int n = inf.readInt(1, MAX_N, "n");
    inf.readEoln();

    // 2. Read the n space-separated array values (a_i)
    for (int i = 1; i <= n; i++) {
        inf.readInt(1, MAX_A, "a_i");
        if (i < n) {
            inf.readSpace();
        }
    }
    inf.readEoln();

    // 3. Read the n-1 edges and validate tree properties
    DSU dsu(n);
    for (int i = 1; i <= n - 1; i++) {
        int u = inf.readInt(1, n, "u");
        inf.readSpace();
        int v = inf.readInt(1, n, "v");
        inf.readEoln();

        // Check for self-loops
        ensuref(u != v, "Self-loops are not allowed: edge between %d and %d", u, v);

        // Ensure no cycles or parallel edges exist
        ensuref(dsu.unite(u, v), "The graph contains a cycle or parallel edges. It is not a valid tree.");
    }

    // Ensure there is no trailing garbage in the input file
    inf.readEof();

    return 0;
}