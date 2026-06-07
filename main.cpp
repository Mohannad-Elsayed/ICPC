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

namespace corasick {
    const int N = 1001000;
    const int SIGMA = 26;

    int nxt[N][SIGMA];
    int fail_link[N];
    int dict_link[N];
    int match_idx[N];
    int nodes;

    void init() {
        nodes = fail_link[0] = dict_link[0] = 0;
        memset(nxt[0], 0, sizeof(nxt[0]));
        match_idx[0] = -1;
        nodes++;
    }

    int create_node() {
        memset(nxt[nodes], 0, sizeof(nxt[nodes]));
        fail_link[nodes] = dict_link[nodes] = 0;
        match_idx[nodes] = -1;
        return nodes++;
    }

    int insert(const string& pattern, int id) {
        int cur = 0;
        for (char c : pattern) {
            if (!nxt[cur][c]) {
                nxt[cur][c] = create_node();
            }
            cur = nxt[cur][c];
        }
        if (~match_idx[cur]) return match_idx[cur];
        return match_idx[cur] = id;
    }

    void build() {
        queue<int> q;

        for (int c = 0; c < SIGMA; ++c) {
            int chi = nxt[0][c];
            if (chi) {
                fail_link[chi] = dict_link[chi] = 0;
                q.push(chi);
            }
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int c = 0; c < SIGMA; ++c) {
                int chi = nxt[u][c];

                if (chi) {
                    fail_link[chi] = nxt[fail_link[u]][c];

                    int fail = fail_link[chi];
                    if (match_idx[fail] != -1)
                        dict_link[chi] = fail;
                    else
                        dict_link[chi] = dict_link[fail];
                    q.push(chi);
                } else
                    nxt[u][c] = nxt[fail_link[u]][c];
            }
        }
    }

    vector<vector<int>> search(const string& text, const vector<int>& lengths) {
        vector<vector<int>> ret(lengths.size());
        for (int cur = 0, i = 0; i < text.length(); ++i) {
            cur = nxt[cur][text[i]];

            for (int u = cur; u; u = dict_link[u]) {
                int id = match_idx[u];
                if (id != -1) {
                    ret[id].push_back(i - lengths[id] + 1);
                }
            }
        }
        return ret;
    }
} /* init() insert() build() search() */

void solve() {

}