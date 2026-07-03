// #define ONLINE_JUDGE
#include "bits/stdc++.h"
using namespace std;
#if !defined(mhnd01s) || defined(ONLINE_JUDGE)
#define print(...) ((void)0)
#endif
using ll = long long;
void solve();
signed main() {
#ifdef mhnd01s
    int x = mt19937(random_device()())()%100;printf("%d\n", x);
    freopen("out", "wt", stdout);
#else
    cin.tie(0)->sync_with_stdio(0);
#endif
    cin.exceptions(cin.failbit);
    int t = 1;
    cin >> t;
    while(t--) {
        solve();
        if(t) cout << '\n';
    }return 0;
}

struct DSU {
    vector<int> p, sz;
    int n, comps;
    DSU(int _n = 0) { init(_n); }
    void init(int _n) {
        n = _n + 10; comps = _n;
        p.resize(n); sz.assign(n, 1);
        iota(p.begin(), p.end(), 0);
    }
    int find(int u) { return u == p[u] ? u : p[u] = find(p[u]); }
    bool unite(int u, int v) {
        u = find(u), v = find(v);
        if (u == v) return 0;
        if (sz[u] < sz[v]) swap(u, v);
        p[v] = u; sz[u] += sz[v]; comps--;
        return 1;
    }
    bool same(int u, int v) { return find(u) == find(v); }
    int size(int u) { return sz[find(u)]; }
    int size() { return comps; }
};

void solve() {
    int n, m; cin >> n >> m;
    vector<int> v(n), frq(m+1);
    for (auto &i : v) cin >> i, frq[i]++;
    vector<vector<int>> idx(m+1);
    for (auto x : v) idx[x].reserve(frq[x]+2);
    vector<array<int, 3>> edges;
    for (int i = 0; i < n; i++) idx[v[i]].push_back(i);

    for (int i = m; i; i--) {
        if (idx[i].empty()) continue;
        for (int j = 1; j < idx[i].size(); j++)
            edges.push_back({-i, idx[i][0], idx[i][j]});
        for (int d = i << 1; d <= m; d+=i)
            for (int j = 0; j < idx[d].size(); j++)
                edges.push_back({-i, idx[i][0], idx[d][j]});
    }

    sort(edges.begin(), edges.end());
    print(edges);

    ll ans = 0, cnt = 0;
    DSU d(n);
    for (auto [w, x, y] : edges) {
        if (d.unite(x, y)) {
            cnt++;
            ans += d.unite(x, y) * w;
        }
    }
    cout << -ans + n - 1 - cnt;
}