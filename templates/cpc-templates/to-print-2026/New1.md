---
title: 11. Ziad.
---

\newpage

# Two Sat
```cpp {.numberLines}
struct two_sat {
    int n, timer, scc_cnt;
    vector<vector<int> > g;
    vector<int> dfn, low, scc, st, answer;
    vector<char> in_st;

    two_sat(int n) : n(n), timer(0), scc_cnt(0) {
        g.assign(2 * n, {});
        dfn.assign(2 * n, 0);
        low.assign(2 * n, 0);
        scc.assign(2 * n, 0);
        in_st.assign(2 * n, false);
        answer.assign(n, 0);
    }

    int var(int i, bool val) { return i + (val ? 0 : n); }

    void add_edge(int u, int v) { g[u].push_back(v); }

    void add_imply(int i, bool f, int j, bool k) {
        add_edge(var(i, f), var(j, k));
    }

    void set_true(int i) { add_imply(i, false, i, true); }
    void set_false(int i) { add_imply(i, true, i, false); }

    void add_or(int i, bool f, int j, bool k) {
        add_imply(i, !f, j, k);
        add_imply(j, !k, i, f);
    }

    void add_xor(int i, bool f, int j, bool k) {
        add_or(i, f, j, k);
        add_or(i, !f, j, !k);
    }

    void add_xnor(int i, bool f, int j, bool k) {
        add_or(i, f, j, !k);
        add_or(i, !f, j, k);
    }

    void add_and(int i, bool f, int j, bool k) {
        add_imply(i, !f, i, f);
        add_imply(j, !k, j, k);
    }

    void dfs(int u) {
        dfn[u] = low[u] = ++timer;
        st.push_back(u);
        in_st[u] = true;
        for (int v: g[u]) {
            if (!dfn[v]) {
                dfs(v);
                low[u] = min(low[u], low[v]);
            } else if (in_st[v]) {
                low[u] = min(low[u], dfn[v]);
            }
        }
        if (low[u] == dfn[u]) {
            scc_cnt++;
            while (true) {
                int v = st.back();
                st.pop_back();
                in_st[v] = false;
                scc[v] = scc_cnt;
                if (u == v) break;
            }
        }
    }

    bool satisfiable() {
        timer = scc_cnt = 0;
        fill(dfn.begin(), dfn.end(), 0);

        for (int i = 0; i < 2 * n; i++)
            if (!dfn[i]) dfs(i);

        for (int i = 0; i < n; i++) {
            if (scc[i] == scc[i + n]) return false;
            answer[i] = scc[i] < scc[i + n];
        }
        // answer[i] = 0 ==> negative
        return true;
    }
};
```
# Min Cost Max Flow
```cpp {.numberLines}
const int inf = 1e9;
class MinCostMaxFlow {
    struct Edge { int to, rev, cap, cost; };
    int n;
    vector<vector<Edge>> g;
    vector<int> dis, pru, pri, h;

public:
    MinCostMaxFlow(int n) : n(n), g(n), dis(n), pru(n), pri(n), h(n) {}

    void addEdge(int u, int v, int cap, int cost) {
        g[u].push_back({v, (int)g[v].size(), cap, cost});
        g[v].push_back({u, (int)g[u].size() - 1, 0, -cost});
    }

    pair<int, int> flow(int s, int t, int maxF = inf) {
        int res = 0, flow = 0;
        fill(h.begin(), h.end(), 0);

        while (maxF > 0) {
            priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> q;
            fill(dis.begin(), dis.end(), inf);
            dis[s] = 0, q.emplace(0, s);

            while (!q.empty()) {
                auto [d, v] = q.top(); q.pop();
                if (dis[v] < d) continue;
                for (int i = 0; i < g[v].size(); i++) {
                    auto& e = g[v][i];
                    if (e.cap > 0 && dis[e.to] > dis[v] + e.cost + h[v] - h[e.to]) {
                        dis[e.to] = dis[v] + e.cost + h[v] - h[e.to];
                        pru[e.to] = v, pri[e.to] = i;
                        q.emplace(dis[e.to], e.to);
                    }
                }
            }
            if(dis[t] == inf) break;

            for(int v = 0; v < n; ++v) h[v] += dis[v];
            int d = maxF;
            for(int v = t; v != s; v = pru[v]) d = min(d, g[pru[v]][pri[v]].cap);

            maxF -= d, flow += d, res += d * h[t];
            for (int v = t; v != s; v = pru[v]) {
                auto& e = g[pru[v]][pri[v]];
                e.cap -= d, g[v][e.rev].cap += d;
            }
        }
        return {flow, res};
    }
};
```
# SoS DP
```cpp {.numberLines}
Computes sum over all subsets (or superset) of bitmask values efficiently using bitwise DP.

f[x]++;

f1[(N - 1) & ~x]++;

number of y such that:

x | y = x => f[x]

x & y = x => f1[(N - 1) & ~x]

x & y = 0 => f[(N - 1) & ~x]

!invert -> sum of subsets, invert -> undo pre operation
void sos(vector<int> &dp, bool invert = false) {
    for (int i = 0; 1 << i < dp.size(); i++) {
        for (int mask = 0; mask < dp.size(); mask++) {
            if (mask & (1 << i)) {
                dp[mask] += invert ? -dp[mask ^ (1 << i)] : dp[mask ^ (1 << i)];
                // dp[mask] >= mod? dp[mask] -= mod: dp[mask] < 0? dp[mask] += mod: 0;
            }
        }
    }
}

template<typename T>
void SubsetZetaTransform(vector<T>& v) {
    const int n = v.size();
    for (int j = 1; j < n; j <<= 1) {
        for (int i = 0; i < n; i++)
            if (i & j)
                v[i] += v[i ^ j];
    }
}

template<typename T>
void SubsetMobiusTransform(vector<T>& v) {
    const int n = v.size();
    for (int j = 1; j < n; j <<= 1) {
        for (int i = 0; i < n; i++)
            if (i & j)
                v[i] -= v[i ^ j];
    }
}

template<typename T>
void SupersetZetaTransform(vector<T>& v) {
    const int n = v.size(); // n must be a power of 2
    for (int j = 1; j < n; j <<= 1) {
        for (int i = 0; i < n; i++)
            if (i & j)
                v[i ^ j] += v[i];
    }
}

template<typename T>
void SupersetMobiusTransform(vector<T>& v) {
    const int n = v.size(); // n must be a power of 2
    for (int j = 1; j < n; j <<= 1) {
        for (int i = 0; i < n; i++)
            if (i & j)
                v[i ^ j] -= v[i];
    }
}

1. Count how many array elements are submasks of x.
   freq[value]++
   SubsetZetaTransform(freq);
   Then:
       freq[x]
   = number of input values y such that
       y ⊆ x


2. Count how many array elements are supermasks of x.
   freq[value]++
   SupersetZetaTransform(freq);
   Then:
       freq[x]
   = number of input values y such that
       y ⊇ x

3. Recover the original array after a transform.
   SubsetZetaTransform(freq);
   SubsetMobiusTransform(freq);
   // freq is back to its original values

4. Disjoint masks.
   x & y == 0
   y must be a submask of ~x.
   For B bits:
       complement = (1 << B) - 1 ^ x
   So after subset-zeta preprocessing:
       count = freq[complement]
```