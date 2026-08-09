---
title: 3. Graphs and Trees
---
# Bellman Ford
```cpp {.numberLines}
// Computes Single-Source Shortest Paths (SSSP) on graphs that 
// contain NEGATIVE edge weights.
// It explicitly detects negative weight cycles reachable from the source vertex.
// - Average Time Complexity: O(E) where E is the number of edges.
// - Worst Case Time Complexity: O(V * E). 
// vector<int> dist;
// bool ok = spfa(0, adj, dist); 

const int INF = 1e9;
bool spfa(int s, const vector<vector<pair<int, int>>>& adj, vector<int>& d) {
    int n = adj.size();
    d.assign(n, INF);
    vector<int> cnt(n, 0);
    vector<int> inqueue(n, 0); 
    queue<int> q;

    d[s] = 0;
    q.push(s);
    inqueue[s] = 1;
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        inqueue[v] = 0;
        for (const auto& [to, len] : adj[v]) {
            if (d[v] + len < d[to]) {
                d[to] = d[v] + len;
                if (!inqueue[to]) {
                    q.push(to);
                    inqueue[to] = 1;
                    if (++cnt[to] > n) return false;  // negative cycle
                }
            }
        }
    }
    return true;
}
```
# Floyd Tricks
```cpp {.numberLines}
void TransitiveClosure(int n, vector<vector<int>>& adj) {
	// 0 = disconnected, 1 = connected
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				adj[i][j] |= (adj[i][k] & adj[k][j]);
}

void minimax(int n, vector<vector<int>>& adj) {
	// Path such that max value on road is minimized
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				adj[i][j] = min(adj[i][j], max(adj[i][k], adj[k][j]));
}

void maximin(int n, vector<vector<int>>& adj) {
	// Path such that min value on road is maximized
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				adj[i][j] = max(adj[i][j], min(adj[i][k], adj[k][j]));
}

void longestPathDAG(int n, vector<vector<int>>& adj) {
	// Only works for DAGs (no positive cycles)
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				adj[i][j] = max(adj[i][j], max(adj[i][k], adj[k][j]));
}

void countPaths(int n, vector<vector<int>>& adj) {
	// Floyd-Warshall for counting number of paths
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				adj[i][j] += adj[i][k] * adj[k][j];
}

bool isNegativeCycle(int n, const vector<vector<int>>& adj) {
	// run floyd first
	for (int i = 0; i < n; ++i)
		if (adj[i][i] < 0)
			return true;
	return false;
}

bool anyEffectiveCycle(int n, const vector<vector<int>>& adj, 
        int src, int dest, int OO) {
	// run floyd first
	for (int i = 0; i < n; ++i)
		if (adj[i][i] < 0 && adj[src][i] < OO && adj[i][dest] < OO)
			return true;
	return false;
}
```
# Topological Sort
```cpp {.numberLines}
queue<int> queue;
for (int i = 0; i < n; i++) 
    if (in_degree[i] == 0) { queue.push(i); }

vector<int> top_sort;
while (!queue.empty()) {
    int curr = queue.front();
    queue.pop();
    top_sort.push_back(curr);
    for (int next : graph[curr]) {
        if (--in_degree[next] == 0) { queue.push(next); }
    }
}
```
# Bipartite Check
```cpp {.numberLines}
int n;
vector<vector<int>> adj;

vector<int> side(n, -1);
bool is_bipartite = true;
queue<int> q;
for (int st = 0; st < n; ++st) {
    if (side[st] == -1) {
        q.push(st);
        side[st] = 0;
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int u : adj[v]) {
                if (side[u] == -1) {
                    side[u] = side[v] ^ 1;
                    q.push(u);
                } else {
                    is_bipartite &= side[u] != side[v];
                }
            }
        }
    }
}
```
# Dijkstra
```cpp {.numberLines}
vector<ll> dijkstra(int src, vector<int> &parent) {
    vector<ll> dist(n + 1, INF);
    parent.assign(n + 1, -1);
    priority_queue<pair<ll, int>, vector<pair<ll, int> >, greater<pair<ll, int> > >
            pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();
        if (d != dist[v]) continue;
        for (auto [u, w]: graph[v]) {
            if (dist[u] > d + w) {
                dist[u] = d + w;
                parent[u] = v;
                pq.push({dist[u], u});
            }
        }
    }
    return dist;
}
```
# Tree Diameter
```cpp {.numberLines}
pair<int, int> dfs(int u, int p) {
    pair<int, int> ret = {0, u};
    for (int v : tree[u]) {
        if (v == p) continue;
        auto x = dfs(v, u);
        ret = max(ret, {x.first + 1, x.second});
    }
    return ret;
}
```
# Tree
```cpp {.numberLines}
struct tree {
    int root = 0;
    vector<vector<int>> g;
    explicit tree(int n) : g(n) { }
    void add(int u, int v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> &operator[](int u) {
        return g[u];
    }

    int cntDfs = 0;
    vector<int> in, out, lvl, sz, top, par, seq;

    void init(int rt = 0) {
        root = rt;
        in = out = lvl = top = par = seq = vector<int>(g.size());
        sz.resize(g.size(), 1);
        par[root] = top[root] = root;
        dfs(root);
        dfs2(root);
    }

    void dfs(int u) {
        for(int &v : g[u]) {
            lvl[v] = lvl[u] + 1;
            par[v] = u;
            g[v].erase(find(g[v].begin(), g[v].end(), u));
            dfs(v);
            sz[u] += sz[v];
            if(sz[v] > sz[g[u][0]])
                swap(v, g[u][0]);
        }
    }
    void dfs2(int u) {
        in[u] = cntDfs++;
        seq[in[u]] = u;
        for(int v : g[u]) {
            top[v] = v == g[u][0]? top[u]: v;
            dfs2(v);
        }
        out[u] = cntDfs - 1;
    }

    int jump(int u, int k) {
        if(k > lvl[u]) return -1;

        int d = lvl[u] - k;
        while (lvl[top[u]] > d) {
            u = par[top[u]];
        }

        return seq[in[u] - lvl[u] + d];
    }

    bool isAncestor(int u, int v) {
        return in[u] <= in[v] && in[v] <= out[u];
    }

    int lca(int u, int v) {
        if(lvl[u] > lvl[v]) swap(u, v);
        if(isAncestor(u, v)) return u;
        while (top[u] != top[v]) {
            if (lvl[top[u]] > lvl[top[v]]) {
                u = par[top[u]];
            } else {
                v = par[top[v]];
            }
        }
        return lvl[u] < lvl[v]? u: v;
    }

    int dis(int u, int v) {
        return lvl[u] + lvl[v] - 2 * lvl[lca(u, v)];
    }
};
```
# Tree Hash (rooted/unrooted)
```cpp {.numberLines}
// tree hash
using u64 = uint64_t;
mt19937_64 rng = []{
    u64 time_entropy = chrono::steady_clock::now().time_since_epoch().count();
    u64 memory_entropy = (uintptr_t)make_unique<char>().get();
    seed_seq ss{time_entropy, memory_entropy};
    return mt19937_64(ss);
}();
u64 SEED = rng();
u64 mix(u64 x) {
    x += SEED + 0x9e3779b97f4a7c15;
    x = (x ^ x >> 30) * 0xbf58476d1ce4e5b9;
    x = (x ^ x >> 27) * 0x94d049bb133111eb;
    return x ^ x >> 31;
}
u64 treehash(int u, int p, vector<vector<int>>& t) {
    u64 ret = 1;
    // do ret = ret * P + mix() if the order is important
    for (int v : t[u]) if (v ^ p)
        ret += mix(treehash(v, u, t));
    return ret;
}
// unrooted: find centroids and treat them as roots
u64 utreehash(vector<vector<int>>& t) {
    int n = t.size();
    vector<int> sz(n), cents;
    function<void(int, int)> dfs1 = [&](int u, int p) {
        sz[u] = 1;
        bool can = 1;
        for (int v : t[u]) if (v ^ p) {
            dfs1(v, u);
            if (sz[v] * 2 > n) can = 0;
            sz[u] += sz[v];
        }
        if (n - sz[u] > n/2) can = 0;
        if (can) cents.push_back(u);
    };
    dfs1(0, 0);
    return min(treehash(cents.front(), -1, t), treehash(cents.back(), -1, t));
}
```
# Binary Lifting
```cpp {.numberLines}
int Log = __lg(n) + 1;
vector<vector<int>> lift(n+1, vector<int>(Log, -1));
function<void(int, int)> dfs2 = [&](int u, int p) {
    for (auto v : tree[u]) if (v != p) {
        lift[v][0] = u;
        for (int l = 1; l < Log && ~lift[v][l-1]; l++)
                lift[v][l] = lift[ lift[v][l-1] ][l-1];
        dfs2(v, u);
    }
};
dfs2(root, -1);

if (depth[u] < depth[v])
    swap(u, v);
int k = depth[u] - depth[v];

for (int l = Log-1; ~l && ~u; l--) {
    if (k >> l & 1)
        u = lift[u][l];
}
if (u == v) return u;
for (int l = Log-1; l > -1; l--) {
    if (lift[u][l] != lift[v][l])
        u = lift[u][l], v = lift[v][l];
}
return lift[u][0];
```
# LCA
```cpp {.numberLines}
struct LCA {
    int n, m, LOG;
    vector<vector<int>> g;
    vector<int> euler, first, depth, lg;
    vector<vector<int>> st;

    LCA(const vector<vector<int>>& tree, int root) {
        n = tree.size();
        g = tree;
        first.assign(n, -1);
        depth.assign(n, 0);

        dfs(root, -1, 0);
        m = euler.size();

        lg.assign(m+1, 0);
        for (int i = 2; i <= m; i++)
            lg[i] = lg[i>>1] + 1;
        LOG = lg[m] + 1;

        st.assign(m, vector<int>(LOG));
        for (int i = 0; i < m; i++)
            st[i][0] = euler[i];

        for (int j = 1; j < LOG; j++) {
            for (int i = 0; i + (1<<j) <= m; i++) {
                int x = st[i][j-1], y = st[i + (1<<(j-1))][j-1];
                st[i][j] = (depth[x] < depth[y] ? x : y);
            }
        }
    }

    void dfs(int u, int p, int d) {
        if (first[u] == -1) first[u] = euler.size();
        euler.push_back(u);
        depth[u] = d;
        for (int v : g[u]) {
            if (v == p) continue;
            dfs(v, u, d+1);
            euler.push_back(u);
        }
    }

    int lca(int u, int v) {
        int L = first[u], R = first[v];
        if (L > R) swap(L, R);
        int len = R - L + 1, k = lg[len];
        int x = st[L][k], y = st[R - (1<<k) + 1][k];
        return depth[x] < depth[y] ? x : y;
    }

    int dis(int u, int v) {
        return depth[u] + depth[v] - 2 * depth[lca(u, v)];
    }
};
```
# DSU on Trees
```cpp {.numberLines}
void pre(int u) {
    sz[u] = 1;
    for (int &v : tree[u]) {
        tree[v].erase(find(tree[v].begin(), tree[v].end(), u));
        pre(v);
        sz[u] += sz[v];
        if (sz[v] > sz[tree[u].front()]) swap(v, tree[u].front());
    }
}

void addver(int u) {
    
}
void removever(int u) {
    
}

void addsub(int u) {
    addver(u);
    for (int v : tree[u]) addsub(v);
}
void removesub(int u) {
    removever(u);
    for (int v : tree[u]) removesub(v);
}

void dfs(int u) {
    for (int v : tree[u]) if (v != tree[u].front()) {
        dfs(v);
        removesub(v);
    }
    if (!tree[u].empty())
        dfs(tree[u].front());
    addver(u);
    for (int v : tree[u])
        if (v != tree[u].front())
            addsub(v);

    // query[u]
}
```
# SCC / Strongly Connected Componenets
```cpp {.numberLines}
// --- USAGE REQUIREMENTS ---
// 1. timer must be declared and initialized to 0 outside the function.
// 2. g is your 0-based directed adjacency list (vector<vector<int>>).
// 3. v is a 0-based array/vector of vertex weights
    (used to find the min weight per SCC).
// 4. idscc[i] will hold the 0-based SCC ID for vertex i.
// 5. cond generates the condensed DAG where each node is a distinct SCC.

// --- MODIFICATIONS FOR BRIDGES & ARTICULATION POINTS (UNDIRECTED GRAPH) ---
// 1. Update the lambda signature to pass the parent: tarj = [&](int u, int p = -1)
// 2. Add int children = 0; at the top of the lambda to 
    count root children for APs.
// 3. Ignore the back-edge to the parent in the loop: if (v_ == p) continue;
// 4. You can completely remove vis, stk, mn, and idscc 
    arrays if you are not extracting SCCs. 

vector<int> tin(n, -1), low(n), vis(n), stk, mn, idscc(n, -1);
function<void(int)> tarj = [&](int u) { // Change to: (int u, int p = -1)
    tin[u] = low[u] = timer++;
    vis[u] = 1; // Marks u as currently in the active SCC stack
    stk.push_back(u);
    
    // int children = 0; // Uncomment for Articulation Points

    for (int v_ : g[u]) {
        // Uncomment to prevent traversing back to parent in undirected graphs
        // if (v_ == p) continue; 

        if (tin[v_] == -1) {
            // children++; // Uncomment for Articulation Points
            tarj(v_); // Change to: tarj(v_, u)
            low[u] = min(low[u], low[v_]);
            
            // --- BRIDGE CHECK ---
            // if (low[v_] > tin[u]) { /* Edge (u, v_) is a bridge */ }
            
            // --- ARTICULATION POINT CHECK (NON-ROOT) ---
            // if (low[v_] >= tin[u] && p != -1) { /* Vertex u is an AP */ }

        } else if (vis[v_]) { 
            // Back-edge found. v_ is already visited AND 
            // still in the current stack
            low[u] = min(low[u], tin[v_]);
        }
    }

    // --- SCC EXTRACTION LOGIC ---
    // If u is the root of an SCC, pop all vertices belonging to 
    // it from the stack
    if (low[u] == tin[u]) {
        int t;
        mn.push_back(1e5); // Initialize the minimum weight for this new SCC
        do {
            t = stk.back();
            stk.pop_back();
            vis[t] = 0; // Mark as removed from the active stack
            // Track min value (requires your v array)
            mn.back() = min(mn.back(), v[t]); 
            // Assign the new SCC ID to vertex t
            idscc[t] = (int)mn.size() - 1;    
        } while (t ^ u);
    }
    
    // --- ARTICULATION POINT CHECK (ROOT) ---
    // if (p == -1 && children > 1) { /* Vertex u (the root) is an AP */ }
};

for (int i = 0; i < n; i++)
    if (tin[i] == -1)
        tarj(i); // Change to: tarj(i, -1)

// --- BUILD THE CONDENSED DAG ---
vector<vector<int>> cond(mn.size());
for (int i = 0; i < n; i++)
    for (int u : g[i])
        if (idscc[i] ^ idscc[u]) // If an edge connects two different SCCs
            cond[idscc[i]].push_back(idscc[u]);
```
# WLCA
```cpp {.numberLines}
template<class T>
struct WLCA {
    int n, Log;
    vector<vector<int>> up;
    vector<vector<T>> val;
    vector<int> lvl;
    explicit WLCA(vector<vector<pair<int, T>>> &g, int root = 0) : n((int)g.size()),
            lvl(n), Log(__lg(n) + 1), up(n, vector<int>(Log, root)), 
            val(n, vector<T>(Log)) {
        function<void(int)> dfs = [&](int u) -> void {
            for(auto [v, w] : g[u]) {
                if(v == up[u][0]) continue;
                lvl[v] = lvl[u] + 1, up[v][0] = u, val[v][0] = w;
                for(int l = 1; l < Log; l++) {
                    up[v][l] = up[up[v][l - 1]][l - 1];
                    val[v][l] = val[v][l - 1] + val[up[v][l - 1]][l - 1];
                }
                dfs(v);
            }
        };
        dfs(root);
    }
    pair<int, T> k_ancestor(int u, int k) {
        T ans;
        while(k) {
            ans = ans + val[u][__builtin_ctz(k)];
            u = up[u][__builtin_ctz(k)];
            k &= k - 1;
        }
        return {u, ans};
    }
    int lca(int u, int v) {
        if(lvl[u] < lvl[v]) swap(u, v);
        u = k_ancestor(u, lvl[u] - lvl[v]).first;
        if(u == v) return u;
        for(int l = Log - 1; l >= 0; l--) {
            if(up[u][l] ^ up[v][l]) {
                u = up[u][l], v = up[v][l];
            }
        }
        return up[u][0];
    }
    int dis(int u, int v, int l = -1) {
        if(l == -1) l = lca(u, v);
        return lvl[u] + lvl[v] - 2 * lvl[l];
    }
};
```
# Rerooter
```cpp {.numberLines}
namespace reroot {
    const auto exclusive = [](const auto& a, const auto& base,
                                const auto& merge_into, int vertex) {
        int n = (int)a.size();
        using Aggregate = decay_t<decltype(base)>;
        vector<Aggregate> b(n, base);
        for (int bit = __lg(n); bit >= 0; --bit) {
            for (int i = n - 1; i >= 0; --i) b[i] = b[i >> 1];
            int sz = n - (n & !bit);
            for (int i = 0; i < sz; ++i) {
                int index = (i >> bit) ^ 1;
                b[index] = merge_into(b[index], a[i], vertex, i);
            }
        }
        return b;
    };
    // MergeInto : Aggregate * Value * Vertex(int) * EdgeIndex(int) -> Aggregate
    // Base : Vertex(int) -> Aggregate
    // FinalizeMerge : Aggregate * Vertex(int) * EdgeIndex(int) -> Value
    const auto rerooter = [](const auto& g, const auto& base, 
                const auto& merge_into, const auto& finalize_merge) {
        int n = (int)g.size();
        using Aggregate = decay_t<decltype(base(0))>;
        using Value = decay_t<decltype(finalize_merge(base(0), 0, 0))>;
        vector<Value> root_dp(n), dp(n);
        vector<vector<Value>> edge_dp(n), redge_dp(n);

        vector<int> bfs, parent(n);
        bfs.reserve(n);
        bfs.push_back(0);
        for (int i = 0; i < n; ++i) {
            int u = bfs[i];
            for (auto v : g[u]) {
                if (parent[u] == v) continue;
                parent[v] = u;
                bfs.push_back(v);
            }
        }

        for (int i = n - 1; i >= 0; --i) {
            int u = bfs[i];
            int p_edge_index = -1;
            Aggregate aggregate = base(u);
            for (int edge_index = 0; edge_index < (int)g[u].size(); ++edge_index) {
                int v = g[u][edge_index];
                if (parent[u] == v) {
                    p_edge_index = edge_index;
                    continue;
                }
                aggregate = merge_into(aggregate, dp[v], u, edge_index);
            }
            dp[u] = finalize_merge(aggregate, u, p_edge_index);
        }

        for (auto u : bfs) {
            dp[parent[u]] = dp[u];
            edge_dp[u].reserve(g[u].size());
            for (auto v : g[u]) edge_dp[u].push_back(dp[v]);
            auto dp_exclusive = exclusive(edge_dp[u], base(u), merge_into, u);
            redge_dp[u].reserve(g[u].size());
            for (int i = 0; i < (int)dp_exclusive.size(); ++i) 
                redge_dp[u].push_back(finalize_merge(dp_exclusive[i], u, i));
            root_dp[u] = finalize_merge(n > 1 ? 
                merge_into(dp_exclusive[0], edge_dp[u][0], u, 0) : 
                base(u), u, -1);
            for (int i = 0; i < (int)g[u].size(); ++i) {
                dp[g[u][i]] = redge_dp[u][i];
            }
        }

        return make_tuple(move(root_dp), move(edge_dp), move(redge_dp));
    };
}  // namespace reroot
// [&](Aggregate agg, Aggregate chdp, int v, int eid)
// [&](Aggregate agg, int v, int eid);
```
# Dynamic Connectivity
```cpp {.numberLines}
struct Query {
    char t;
    int u, v;
};

struct Elem {
    int u, v, szU, cnt;
};

struct DSURollback {
    // offline : [+ a b] add edge between a, b
    //           [- a b] remove edge between a, b
    //           [?] number of connected components
    int cnt;
    stack<Elem> st;
    vector<bool> ans;
    vector<int> sz, par;
    map<int, vector<pair<int, int>>> g;

    DSURollback(int n) {
        cnt = n;
        par.resize(n + 1);
        sz.resize(n + 1, 1);
        iota(par.begin(), par.end(), 0);
    }

    void rollback(int x) {
        while (st.size() > x) {
            auto e = st.top();
            st.pop();
            cnt = e.cnt;
            sz[e.u] = e.szU;
            par[e.v] = e.v;
        }
    }

    int findSet(int u) {
        return par[u] == u ? u : findSet(par[u]);
    }

    void update(int u, int v) {
        st.push({u, v, sz[u], cnt});
        cnt--;
        par[v] = u;
        sz[u] += sz[v];
    }

    void unionSet(int u, int v) {
        u = findSet(u);
        v = findSet(v);
        if (u != v) {
            if (sz[u] < sz[v])
                swap(u, v);
            update(u, v);
        }
    }

    void solve(int x, int l, int r) {
        int cur = st.size();

        for (auto i: g[x])
            unionSet(i.first, i.second);

        if (l == r) {
            if (ans[l])
                cout << cnt << endl;
            rollback(cur);
            return;
        }
        int m = (l + r) >> 1;
        solve(x * 2, l, m);
        solve(x * 2 + 1, m + 1, r);
        rollback(cur);
    }

    void traverse(int x, int lX, int rX, int l, int r, int u, int v) {
        if (rX < l || lX > r)
            return;
        if (lX >= l && rX <= r) {
            g[x].emplace_back(u, v);
            return;
        }
        int m = (lX + rX) >> 1;
        traverse(x * 2, lX, m, l, r, u, v);
        traverse(x * 2 + 1, m + 1, rX, l, r, u, v);
    }

    void build(vector<Query> &queries) {
        int q = queries.size();
        ans.resize(q);
        map<pair<int, int>, vector<pair<int, int>>> mp;
        for (int i = 0; i < queries.size(); i++) {
            auto cur = queries[i];
            if (cur.u > cur.v)
                swap(cur.u, cur.v);
            if (cur.t == '?')
                ans[i] = 1;
            else if (cur.t == '+')
                mp[{cur.u, cur.v}].emplace_back(i, queries.size());
            else {
                mp[{cur.u, cur.v}].back().second = i - 1;
                traverse(1, 0, q - 1, mp[{cur.u, cur.v}].back().first, 
                mp[{cur.u, cur.v}].back().second, cur.u, cur.v);
            }
        }

        for (auto i: mp) {
            for (auto j: i.second) {
                if (j.second == q)
                    traverse(1, 0, q - 1, j.first, q - 1, i.first.first, 
                    i.first.second);
            }
        }
    }
};

void testCase() {
    int n, q;
    cin >> n >> q;
    if (!q)
        return;
    DSURollback dsu(n);
    vector<Query> queries(q);
    char t;
    int x, y;
    for (int i = 0; i < q; i++) {
        cin >> t;
        if (t == '?')
            queries[i] = {t, 0, 0};
        else {
            cin >> x >> y;
            if (y < x)
                swap(x, y);
            queries[i] = {t, x, y};
        }
    }

    dsu.build(queries);
    dsu.solve(1, 0, q - 1);
}
```
# Max Flow (Dinik)
```cpp {.numberLines}
class Dinic { // O(n^2 * m), 0-based
   private:
    struct E {
        int to, rev;
        ll c, oc;
        ll f() { return max(oc - c, 0LL); }
    };
    vector<int> lvl, ptr, q;
    vector<vector<E>> g;
    ll dfs(int v, int t, ll f) {
        if (v == t || !f) return f;
        for (int &i = ptr[v]; i < g[v].size(); ++i) {
            E &e = g[v][i];
            if (lvl[e.to] == lvl[v] + 1 && e.c)
                if (ll p = dfs(e.to, t, min(f, e.c))) {
                    e.c -= p;
                    g[e.to][e.rev].c += p;
                    return p;
                }
        }
        return 0;
    }

   public:
    Dinic(int n) : lvl(n), ptr(n), q(n), g(n) {}
    void add_edge(int u, int v, ll c) { // directed
        g[u].push_back({v, (int)g[v].size(), c, c});
        g[v].push_back({u, (int)g[u].size() - 1, 0, 0});
    }
    ll calc(int s, int t) { // source to destination
        ll flow = 0;
        while (true) {
            fill(lvl.begin(), lvl.end(), 0);
            int qs = 0, qe = 0;
            lvl[q[qe++] = s] = 1;
            while (qs < qe && !lvl[t]) {
                int v = q[qs++];
                for (auto &e : g[v])
                    if (!lvl[e.to] && e.c) lvl[q[qe++] = e.to] = lvl[v] + 1;
            }
            if (!lvl[t]) break;
            fill(ptr.begin(), ptr.end(), 0);
            while (ll p = dfs(s, t, LLONG_MAX)) flow += p;
        }
        return flow;
    }
    void reset() { // before calling calc again
        for (auto &adj : g)
            for (auto &e : adj) e.c = e.oc;
    }
    bool leftOfMinCut(int a) { return lvl[a] != 0; }
};
```
# Max Bipartite Matching (Karp) [with building]
```cpp {.numberLines}
struct matching {
    int nl, nr;
    vector<vector<int>> g;
    vector<int> dis, ml, mr;
    explicit matching(int nl, int nr) : nl(nl), nr(nr), g(nl), dis(nl), ml(nl, -1), 
    mr(nr, -1) { }

    void add(int l, int r) { // [i, j]
        g[l].push_back(r);
    }

    void bfs() {
        queue<int> q;
        for(int u = 0; u < nl; u++) {
            if(ml[u] == -1) q.push(u), dis[u] = 0;
            else dis[u] = -1;
        }
        while(!q.empty()) {
            int l = q.front(); q.pop();
            for(int r : g[l]) {
                if(mr[r] != -1 && dis[mr[r]] == -1)
                    q.push(mr[r]), dis[mr[r]] = dis[l] + 1;
            }
        }
    }

    bool canMatch(int l) {
        for(int r : g[l]) if(mr[r] == -1)
            return mr[r] = l, ml[l] = r, true;
        for(int r : g[l]) if(dis[l] + 1 == dis[mr[r]] && canMatch(mr[r]))
            return mr[r] = l, ml[l] = r, true;
        return false;
    }

    int maxMatch() {
        int ans = 0, turn = 1;
        while(turn) {
            bfs(), turn = 0;
            for(int l = 0; l < nl; l++) if(ml[l] == -1)
                turn += canMatch(l);
            ans += turn;
        }
        return ans;
    }

    pair<vector<int>, vector<int>> minCover() {
        vector<int> L, R;
        for (int u = 0; u < nl; ++u) {
            if(dis[u] == -1) L.push_back(u);
            else if(ml[u] != -1) R.push_back(ml[u]);
        }
        return {L, R};
    }
};
```
