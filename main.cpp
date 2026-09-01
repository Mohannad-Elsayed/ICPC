#include <bits/stdc++.h>
using namespace std;

const int MAXN = 100005;
const int MAXA = 100005;

int n, a[MAXN], maxAVal;
vector<int> adj[MAXN], children[MAXN];
int tin[MAXN], tout[MAXN], nodeAtTime[MAXN], timer_, subtreeSz[MAXN];
long long gcdSub[MAXN], gcdComp[MAXN];
int globalCnt[MAXA], degree_[MAXN];

// Compute GCD sum from frequency array using divisor counting
long long computeGcdSum(const vector<int>& cnt, int maxVal) {
    if (maxVal <= 0) return 0;

    // For each d, count how many values are divisible by d
    vector<int> divCnt(maxVal + 1, 0);
    for (int d = 1; d <= maxVal; d++)
        for (int m = d; m <= maxVal; m += d)
            divCnt[d] += cnt[m];

    // Compute exact number of pairs with gcd = d using inclusion-exclusion
    vector<long long> exact(maxVal + 1, 0);
    for (int d = maxVal; d >= 1; d--) {
        exact[d] = (long long)divCnt[d] * (divCnt[d] + 1) / 2;
        for (int m = 2 * d; m <= maxVal; m += d)
            exact[d] -= exact[m];
    }

    // Sum up d * exact_pairs(d)
    long long result = 0;
    for (int d = 1; d <= maxVal; d++)
        result += (long long)d * exact[d];
    return result;
}

// DFS to compute Euler tour and tree structure (rooted at 1)
void dfs(int node, int par) {
    tin[node] = ++timer_;
    nodeAtTime[timer_] = node;
    for (int nb : adj[node])
        if (nb != par) {
            children[node].push_back(nb);
            dfs(nb, node);
        }
    tout[node] = timer_;
}

// Compute subtree frequency arrays using small-to-large merging
// Also computes gcdSub and gcdComp for each node
vector<int>* computeFreq(int node) {
    subtreeSz[node] = 1;
    vector<int>* cnt = new vector<int>(a[node] + 1, 0);
    (*cnt)[a[node]] = 1;

    for (int child : children[node]) {
        vector<int>* childCnt = computeFreq(child);

        // Small-to-large: ensure cnt points to the larger map
        if (cnt->size() < childCnt->size())
            swap(cnt, childCnt);

        int childMax = childCnt->size() - 1;
        if ((int)cnt->size() - 1 < childMax)
            cnt->resize(childMax + 1, 0);

        // Merge child's frequencies into parent
        for (int v = 0; v <= childMax; v++)
            (*cnt)[v] += (*childCnt)[v];
        subtreeSz[node] += subtreeSz[child];
        delete childCnt;
    }

    int maxV = cnt->size() - 1;
    gcdSub[node] = computeGcdSum(*cnt, maxV);

    // Compute complement frequency (global - subtree)
    vector<int> compFreq(maxAVal + 1, 0);
    for (int i = 0; i <= maxAVal; i++)
        compFreq[i] = globalCnt[i] - (i < (int)cnt->size() ? (*cnt)[i] : 0);
    gcdComp[node] = computeGcdSum(compFreq, maxAVal);

    return cnt;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        maxAVal = max(maxAVal, a[i]);
    }
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    if (n == 1) { cout << 0 << endl; return 0; }

    // Root tree at 1 and compute Euler tour
    dfs(1, 0);

    // Compute global frequency array
    memset(globalCnt, 0, sizeof(globalCnt));
    for (int i = 1; i <= n; i++)
        globalCnt[a[i]]++;

    // Compute subtree frequencies and GCD sums
    vector<int>* rootFreq = computeFreq(1);
    delete rootFreq;

    // Compute degrees
    for (int i = 1; i <= n; i++)
        degree_[i] = adj[i].size();

    // Use difference array to compute contributions
    // For node u with degree > 1:
    //   - If root r is outside subtree(u): contribution = gcdSub[u]
    //   - If root r is in subtree(c) for child c of u: contribution = gcdComp[c]
    vector<long long> diff(n + 2, 0);
    for (int u = 1; u <= n; u++) {
        if (degree_[u] <= 1) continue;

        // Add gcdSub[u] for roots outside subtree(u)
        if (tin[u] > 1) { diff[1] += gcdSub[u]; diff[tin[u]] -= gcdSub[u]; }
        if (tout[u] < n) { diff[tout[u] + 1] += gcdSub[u]; diff[n + 1] -= gcdSub[u]; }

        // Add gcdComp[c] for roots in subtree(c)
        for (int c : children[u]) {
            diff[tin[c]] += gcdComp[c];
            diff[tout[c] + 1] -= gcdComp[c];
        }
    }

    // Compute prefix sums and map back to nodes
    vector<long long> results(n + 1);
    long long cur = 0;
    for (int i = 1; i <= n; i++) {
        cur += diff[i];
        results[nodeAtTime[i]] = gcdSub[1] + cur;  // gcdSub[1] = GCD_SUM of entire tree
    }

    for (int i = 1; i <= n; i++) {
        if (i > 1) cout << " ";
        cout << results[i];
    }
    cout << endl;

    return 0;
}