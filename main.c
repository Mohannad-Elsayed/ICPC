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
    // cin >> t;
    while(t--) {
        solve();
        if(t) cout << '\n';
    }return 0;
}

int ans[5050];

int N, M;
const int inf = 1010;
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
            int d = maxF, u1 = pru[t], u2;
            for(int v = t; v != s; v = pru[v]) d = min(d, g[pru[v]][pri[v]].cap), u2 = v;
            ans[u2] = u1;
            maxF -= d, flow += d, res += d * h[t];
            for (int v = t; v != s; v = pru[v]) {
                auto& e = g[pru[v]][pri[v]];
                e.cap -= d, g[v][e.rev].cap += d;
            }
        }
        return {flow, res};
    }
};



void solve() {
    int n, m; cin >> n >> m;
    N = n, M = m;
    vector<int> deps(n), build_cap(m), build_cost(m);
    for (auto &i : deps) cin >> i;
    for (auto &i : build_cap) cin >> i;
    for (auto &i : build_cost) cin >> i;

    MinCostMaxFlow mcmf(n+m+2);
    for (int i = 1; i <= n; i++) mcmf.addEdge(0, i, 1, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (build_cap[j] >= deps[i])
                mcmf.addEdge(i+1, n+1+j, 1, build_cost[j]);
    for (int i = n+1; i <= n+m; i++) mcmf.addEdge(i, n+m+1, 1, 0);

    auto [cost, flow] = mcmf.flow(0, n+m+1, n);
    if (flow < n) return void(cout << "impossible");
    for (int i = 1; i <= n; i++) cout << ans[i]-n << ' ';
}