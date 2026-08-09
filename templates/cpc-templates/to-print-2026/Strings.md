---
title: 5. Strings
---
# Trie (s)
## Global Arrays
```cpp {.numberLines}
const int MXN = 1000005, S = 26, OFFSET = 'a';
int nxt[MXN][S], cnt[MXN], isend[MXN], nodes = 1;
// clear the arrays you add here in clear() function
void clear() {
    for (int i = 0; i < nodes; i++) {
        cnt[i] = isend[i] = 0;
        memset(nxt[i], 0, sizeof(nxt[i]));
    }
    nodes = 1;
}
void insert(const string& s) {
    int u = 0;
    for (char c : s) {
        int v = c - OFFSET;
        if (!nxt[u][v]) nxt[u][v] = nodes++;
        u = nxt[u][v];
        cnt[u]++;
    }
    isend[u]++;
}
int search(const string& s) {
    int u = 0;
    for (char c : s) {
        int v = c - OFFSET;
        if (!nxt[u][v]) return false;
        u = nxt[u][v];
    }
    return cnt[u];
}
// clear() insert() 
```
## Unordered_map
```cpp {.numberLines}
const int MXN = 1000005;
struct Node {
    unordered_map<char, int> nxt;
    int cnt = 0;
} tree[MXN];
int node_cnt = 1;
void clear() {
    for (int i = 0; i < node_cnt; i++) {
        tree[i].nxt.clear();
        tree[i].cnt = 0;
    }
    node_cnt = 1;
}
void insert(const string& s) {
    int u = 0;
    for (char c : s) {
        auto it = tree[u].nxt.find(c);
        if (it == tree[u].nxt.end()) {
            tree[u].nxt[c] = node_cnt;
            u = node_cnt++;
        } else {
            u = it->second;
        }
        tree[u].cnt++;
    }
}
int search(const string& s) {
    int u = 0;
    for (char c : s) {
        auto it = tree[u].nxt.find(c);
        if (it == tree[u].nxt.end()) return false;
        u = it->second;
    }
    return tree[u].cnt;
}
// clear() insert() 
```
## Forward Start Trie
```cpp {.numberLines}
const int MXN = 1000005;
int head[MXN]; // points to the first child of a node
int nxt[MXN];  // points to the next sibling of a node
char val[MXN]; // the character that leads to this node
int cnt[MXN], isend[MXN];
int node_cnt = 1;
void clear() {
    fill(head, head + node_cnt, 0);
    fill(nxt, nxt + node_cnt, 0);
    fill(cnt, cnt + node_cnt, 0);
    fill(isend, isend + node_cnt, 0);
    node_cnt = 1;
}
void insert(const string& s) {
    int u = 0;
    for (char c : s) {
        int child = head[u];
        bool found = false;
        while (child) {
            if (val[child] == c) {
                u = child;
                found = true;
                break;
            }
            child = nxt[child];
        }
        if (!found) {
            val[node_cnt] = c;
            nxt[node_cnt] = head[u];
            head[u] = node_cnt;
            u = node_cnt++;
        }
        cnt[u]++;
    }
}
int search(const string& s) {
    int u = 0;
    for (char c : s) {
        int child = head[u];
        bool found = false;
        while (child) {
            if (val[child] == c) {
                u = child;
                found = true;
                break;
            }
            child = nxt[child];
        }
        if (!found) return 0;
    }
    return cnt[u];
}
```
# Rolling Hash
```cpp {.numberLines}
using u64 = uint64_t;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count() ^ 
    (uintptr_t)make_unique<char>().get());

struct hash61 {
    static const u64 md = (1LL << 61) - 1;
    inline static u64 step = (md >> 2) + rng() % (md >> 1);
    inline static vector<u64> pw = {1};

    int n;
    vector<u64> pref, suff;

    u64 add(u64 a, u64 b) const { return (a += b) >= md ? a - md : a; }
    u64 sub(u64 a, u64 b) const { return (a += md - b) >= md ? a - md : a; }
    u64 mul(u64 a, u64 b) const { return __uint128_t(a) * b % md; }

    template<class T>
    hash61(const T& s) : n(s.size()), pref(n + 1), suff(n + 1) {
        while (pw.size() <= n) pw.push_back(mul(pw.back(), step));
        pref[0] = suff[n] = 1;
        for (int i = 0; i < n; i++) pref[i + 1] = 
            add(mul(pref[i], step), s[i]);
        for (int i = n - 1; i >= 0; i--) suff[i] = 
            add(mul(suff[i + 1], step), s[i]);
    }

    u64 operator()(int l, int r) const { return sub(pref[r + 1], 
        mul(pref[l], pw[r - l + 1])); }
    u64 rev(int l, int r) const { return sub(suff[l],
        mul(suff[r + 1], pw[r - l + 1])); }
};

struct hash61 {
    static const u64 md = (1LL << 61) - 1;
    inline static u64 step = (md >> 2) + rng() % (md >> 1);

    inline static u64 power(u64 b, u64 e) {
        u64 r = 1;
        while (e) {
            if (e & 1) r = (unsigned __int128)r * b % md;
            b = (unsigned __int128)b * b % md;
            e >>= 1;
        }
        return r;
    }

    inline static u64 inv_step = power(step, md - 2);
    inline static vector<u64> pw = {1}, ipw = {1};

    int n = 0;
    vector<u64> pref = {1}, suff = {0};

    u64 add(u64 a, u64 b) const { return (a += b) >= md ? a - md : a; }
    u64 sub(u64 a, u64 b) const { return (a += md - b) >= md ? a - md : a; }
    u64 mul(u64 a, u64 b) const { return (unsigned __int128)a * b % md; }

    hash61() {}
    template<class T> hash61(const T& s) { for (auto c : s) push_back(c); }

    void push_back(u64 c) {
        while (pw.size() <= n + 1) {
            pw.push_back(mul(pw.back(), step));
            ipw.push_back(mul(ipw.back(), inv_step));
        }
        pref.push_back(add(mul(pref.back(), step), c));
        suff.push_back(add(suff.back(), mul(c, pw[n])));
        n++;
    }

    void pop_back() {
        if (!n) return;
        pref.pop_back();
        suff.pop_back();
        n--;
    }

    u64 operator()(int l, int r) const { return sub(pref[r + 1], mul(pref[l], 
        pw[r - l + 1])); }
    u64 rev(int l, int r) const { return mul(sub(suff[r + 1], suff[l]), ipw[l]); }
};

```
# KMP
```cpp {.numberLines}
vector<int> KMP(const string &s){
    int n = s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++){
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]){
            j = pi[j - 1];
        }
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }
    return pi;
}
vector<int> find_occurrences(const string &pat, const string &text){
    string s = pat + '#' + text;
    vector<int> pi = KMP(s), ret;
    int m = pat.size();
    for (int i = m + 1; i < (int)s.size(); i++){
        if (pi[i] == m) ret.push_back(i - 2 * m);
    }
    return ret;
}
vector<vector<int>> automaton(string s){ // aut[s.length() + 1][26]
    s += '#';
    int n = s.length();
    vector<int> pi = KMP(s);
    vector<vector<int>> aut(n, vector<int>(26));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < 26; j++){
            if (i > 0 && 'a' + j != s[i]){
                aut[i][j] = aut[pi[i - 1]][j];
            }else{
                aut[i][j] = i + ('a' + j == s[i]);
            }
        }
    }
    return aut;
}
```
# Z-Algo
```cpp {.numberLines}
vector<int> z_algo(string s) {
    vector<int> z(s.size());
    for(int i = 1, l = 0, r = 0; i < s.size(); i++) {
        if(i < r) z[i] = min(r - i, z[i - l]);
        while(i + z[i] < s.size() && s[z[i]] == s[z[i] + i]) z[i]++;
        if(i + z[i] > r) r = i + z[i], l = i;
    }
    return z;
}
```
# Largest Lexicographical Substring
```cpp {.numberLines}
string largestLexSubstring(const string &s) {
    int n = int(s.size());
    int i = 0, j = 1, k = 0;

    while (j + k < n) {
        if (s[i + k] == s[j + k]) k++;
        else if (s[i + k] < s[j + k])
            /* change it to > if you want lowest */
            i = max(i + k + 1, j), j = i + 1, k = 0; 
        else j = j + k + 1, k = 0;
    }

    return s.substr(i);
}
```
# Manacher
```cpp {.numberLines}
auto manacher(const string &t) {
    string s = "%#";
    s.reserve(t.size() * 2 + 3);
    for(char c : t) s += c + "#"s;
    s += '$';
    // t = aabaacaabaa -> s = %#a#a#b#a#a#c#a#a#b#a#a#$

    vector<int> res(s.size());
    for(int i = 1, l = 1, r = 1; i < s.size(); i++) {
        res[i] = max(0, min(r - i, res[l + r - i]));
        while(s[i + res[i]] == s[i - res[i]]) res[i]++;
        if(i + res[i] > r) {
            l = i - res[i];
            r = i + res[i];
        }
    }
    for(auto &i : res) i--;
    return vector(res.begin() + 2, res.end() - 2); // a#a#b#a#a#c#a#a#b#a#a
    // get max odd len = res[2 * i]; aba -> i = b
    // get max even len = res[2 * i + 1]; abba -> i = first b
}
```
# Aho Corasick Algorithm
```cpp {.numberLines}
namespace corasick {
    const int N = ++++++;
    const int SIGMA = ++++++;

    int nxt[N][SIGMA], fail_link[N], dict_link[N], match_idx[N], nodes;

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
```
# Dynamic Aho Corasick
```cpp {.numberLines}
struct AC {
    int N, P;
    static const int A = 26;
    vector<array<int, A> > next;
    vector<int> link, out_link, cnt;
    vector<vector<int> > out, rev;
    AC() : N(0), P(0) { node(); }

    int node() {
        next.push_back(array<int, A>{0});
        rev.push_back({});
        cnt.emplace_back(0);
        link.emplace_back(0);
        out_link.emplace_back(0);
        out.emplace_back();
        return N++;
    }

    inline int get(char c) { return c - 'a'; }

    int add_pattern(const string &T) {
        int u = 0;
        for (auto c: T) {
            if (!next[u][get(c)]) next[u][get(c)] = node();
            u = next[u][get(c)];
        }
        out[u].push_back(P);
        cnt[u]++;
        return P++;
    }

    void build() {
        queue<int> q;
        q.push(0);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int c = 0; c < A; c++) {
                int v = next[u][c];
                if (!v) next[u][c] = next[link[u]][c];
                else {
                    link[v] = u ? next[link[u]][c] : 0;
                    rev[link[v]].push_back(v);
                    cnt[v] += cnt[link[v]];
                    out_link[v] = out[link[v]].empty() ?
                        out_link[link[v]] : link[v];
                    q.push(v);
                }
            }
        }
    }

    int advance(int u, char c) {
        return next[u][get(c)];
    }

    vector<vector<int> > search(const string &s, vector<int> &pat) {
        int n = (int) s.length(), m = (int) pat.size();
        int u = 0;
        vector<vector<int> > ret(m);
        for (int i = 0; i < n; i++) {
            char c = s[i];
            u = advance(u, c);
            int x = u;
            while (x != 0) {
                for (int v: out[x]) ret[v].push_back(i - pat[v] + 1);
                x = out_link[x];
            }
        }
        return ret;
    }
};

struct dynamic_AC {
    int lg;
    vector<vector<string> > a;
    vector<AC> aho;

    dynamic_AC(int LOG = 20) {
        lg = LOG;
        a = vector<vector<string> >(lg);
        aho = vector<AC>(lg);
    }

    void insert(const string &s) {
        vector<string> have = {s};
        for (int i = 0; i < lg; i++) {
            if (i == lg - 1) {
                lg++;
                a.push_back({});
                aho.push_back(AC());
            }
            if (a[i].empty()) {
                swap(have, a[i]);
                for (auto &v: a[i]) aho[i].add_pattern(v);
                aho[i].build();
                break;
            } else {
                have.insert(end(have), begin(a[i]), end(a[i]));
                vector<string>().swap(a[i]);
                aho[i] = AC();
            }
        }
    }

    ll cnt(const string &s) {
        ll answer = 0;
        for (int i = 0; i < lg; i++) {
            int state = 0;
            for (auto &v: s) {
                state = aho[i].advance(state, v);
                answer += aho[i].cnt[state];
            }
        }
        return answer;
    }
};

struct FullyDynamicAC {
    dynamic_AC added, removed;
    
    // Optional: Protects against deleting a string that doesn't exist.
    // If the problem guarantees valid deletions, you can remove this map.
    unordered_map<string, int> active_freq;

    void insert(const string& s) {
        added.insert(s);
        active_freq[s]++;
    }

    void erase(const string& s) {
        if (active_freq[s] > 0) {
            removed.insert(s);
            active_freq[s]--;
        }
    }

    long long cnt(const string& s) {
        return added.cnt(s) - removed.cnt(s);
    }
};
```
# Suffix Array
```cpp {.numberLines}
// O(n log(n))
struct suffix {
    int n;
    vector<int> p, c, lcp;
    string s;

    explicit suffix(string _s) : n(int(_s.size()) + 1), s(move(_s)), p(n), 
        c(n), lcp(n - 1) {
        s += char(0);
        iota(p.begin(), p.end(), 0);
        sort(p.begin(), p.end(), [&](int i, int j) { return s[i] < s[j]; });
        for (int i = 1; i < n; i++) c[p[i]] = c[p[i - 1]] + (s[p[i]] != s[p[i - 1]]);
        vector<int> nc(n), np(n);
        int k = 1;
        while (k < n) {
            vector<int> f(n + 1);
            for (int i = 0; i < n; i++) p[i] = (p[i] - k + n) % n, f[c[i] + 1]++;
            for (int i = 1; i <= n; i++) f[i] += f[i - 1];
            for(int i = 0; i < n; i++) np[f[c[p[i]]]++] = p[i];
            swap(p, np), nc[p[0]] = 0;
            for (int i = 1; i < n; i++)
                nc[p[i]] = nc[p[i - 1]] +
                           (c[p[i]] != c[p[i - 1]]
                            || c[(p[i] + k) % n] != c[(p[i - 1] + k) % n]);
            swap(c, nc), k <<= 1;
        }
        for(int i = k = 0; i < n - 1; i++) {
            int j = p[c[i] - 1];
            for(; s[i + k] == s[j + k]; k++);
            if(c[i]) lcp[c[i] - 1] = k;
            k = max(k - 1, 0);
        }
        s.pop_back(), c.pop_back(), n--;
        p.erase(p.begin()), lcp.erase(lcp.begin());
        for(auto &i : c) i--;
    }
    vector<vector<int>> table;
    void buildLcp() {
        int LOG = __lg(n) + 1;
        table.resize(LOG, vector<int>(n));
        table[0] = lcp;
        for(int l = 1; l < LOG; l++) {
            for(int i = 0; i + (1 << (l - 1)) < n; i++) {
                table[l][i] = min(table[l - 1][i], table[l - 1][i + (1 << (l - 1))]);
            }
        }
    }
    int query(int l, int r) { // 0-based
        if(l == r) return n - 1 - l;
        l = c[l], r = c[r];
        if(l > r) swap(l, r);
        r--;
        int len = __lg(r - l + 1);
        return min(table[len][l], table[len][r - (1 << len) + 1]);
    }
    auto count(string const &t) { // O(log(n) * size(t))
        int l = int(lower_bound(p.begin(), p.end(), -1, [&](int i, int j) {
            return s.substr(i, min<size_t>(t.size(), n - i)) < t;
        }) - p.begin());
        int r = int(lower_bound(p.begin() + l, p.end(), -1, [&](int i, int j) {
            return s.substr(i, min<size_t>(t.size(), n - i)) <= t;
        }) - p.begin()) - 1;
        return pair{l, r};
    }
};
```
# Suffix Automaton
```cpp {.numberLines}
namespace sam {
    const int N = 4000200, SIGMA = 26; // N two times the max length
    int sa[N][SIGMA], len[N], fail[N], terminal[N], nodes, lst, C[N];

    void init() {
        nodes = lst = 0;
        memset(sa[0], -1, sizeof sa[0]);
        fail[0] = -1;
        len[0] = terminal[0] = 0;
    }

    int create_node() {
        nodes++;
        memset(sa[nodes], -1, sizeof sa[nodes]);
        terminal[nodes] = 0;
        return nodes;
    }

    void insert(char ch) {
        int u = create_node();
        len[u] = len[lst] + 1;
        int p, q;
        for (p = lst; ~p; p = fail[p]) {
            q = sa[p][ch];
            if (~sa[p][ch]) break;
            sa[p][ch] = u;
        }

        if (p == -1) {
            fail[u] = 0;
        } else if (len[p] + 1 == len[q]) {
            fail[u] = q;
        } else {
            int x = create_node();
            len[x] = len[p] + 1;
            memcpy(sa[x], sa[q], sizeof sa[0]);
            for (int v = p; ~v && sa[v][ch] == q; v = fail[v])
                sa[v][ch] = x;
            fail[x] = fail[q];
            fail[q] = fail[u] = x;
        }
        lst = u;
        C[u] = 1;
    }

    void insert(const string &s) {
        for (auto ch: s)
            insert(ch);
    }

    string walk(const string &s) {
        int cur = 0;
        for (auto ch: s) {
            if (sa[cur][ch] == -1) return "NO"s;
            cur = sa[cur][ch];
        }
        return "YES";
    }

    void pre_count() {
        vector<vector<int> > vlen(N);
        for (int i = 0; i < N; i++) vlen[len[i]].push_back(i);
        for (int l = N - 1; ~l; l--)
            for (auto x: vlen[l])
                C[fail[x]] += C[x];
    }

    int count(const string &s) {
        int cur = 0;
        for (auto ch: s) {
            if (!~sa[cur][ch]) return 0;
            cur = sa[cur][ch];
        }
        return C[cur];
    }

    vector<ll> count_distinct() {
        vector<ll> dp(nodes + 1), outdeg(nodes + 1);
        vector<vector<int> > indeg(nodes + 1);
        for (int i = 0; i <= nodes; i++)
            for (int j = 0; j < SIGMA; j++)
                if (~sa[i][j]) {
                    outdeg[i]++;
                    indeg[sa[i][j]].push_back(i);
                }
        queue<int> q;
        for (int i = 0; i <= nodes; i++)
            if (!outdeg[i])
                q.push(i);
        while (!q.empty()) {
            int top = q.front();
            q.pop();
            dp[top] = 1;
            for (int i = 0; i < SIGMA; i++)
                if (~sa[top][i])
                    dp[top] += dp[sa[top][i]];
            for (auto x: indeg[top]) {
                outdeg[x]--;
                if (!outdeg[x]) q.push(x);
            }
        }

        return dp;
    }

    ll count_distinct2() {
        ll ret = 0;
        for (int i = 1; i <= nodes; i++)
            ret += len[i] - len[fail[i]];
        return ret;
    }

    ll sum_distinct() {
        vector<ll> d(nodes + 1), ans(nodes + 1), outdeg(nodes + 1);
        vector<vector<int> > indeg(nodes + 1);
        queue<int> q;
        for (int i = 0; i <= nodes; i++) {
            for (int j = 0; j < SIGMA; j++)
                if (~sa[i][j]) {
                    outdeg[i]++;
                    indeg[sa[i][j]].push_back(i);
                }
            if (!outdeg[i]) q.push(i);
        }
        while (!q.empty()) {
            int top = q.front();
            q.pop();

            d[top] = 1;
            for (int i = 0; i < SIGMA; i++)
                if (~sa[top][i]) {
                    d[top] += d[sa[top][i]];
                    ans[top] += d[sa[top][i]] + ans[sa[top][i]];
                }

            for (int x: indeg[top])
                if (!--outdeg[x])
                    q.push(x);
        }
        return ans.front();
    }

    ll sum_distinct2() {
        ll ret = 0;
        for (int i = 1; i <= nodes; i++) {
            int a = len[i], b = len[fail[i]] + 1;
            ll num = a-b+1;
            ret += num * (a+b)/2;
        }
        return ret;
    }

    string get_kth(ll k) {
        auto d = count_distinct();
        string ret;
        int cur = 0;
        while (k) {
            for (int i = 0; i < SIGMA; i++) if (~sa[cur][i]) {
                if (d[sa[cur][i]] >= k) {
                    k--;
                    ret.push_back(i);
                    cur = sa[cur][i];
                    break;
                }
                k -= d[sa[cur][i]];
            }
        }
        return ret;
    }

    string smallest_shift(int n) {
        int cur = 0;
        string ret;
        while (n--) {
            for (int i = 0; i < SIGMA; i++) if (~sa[cur][i]) {
                cur = sa[cur][i];
                ret.push_back(i);
                break;
            }
        }
        return ret;
    }

    vector<int> vis(N);
    void dfs(int cur, const vector<ll>& d) {
        if (vis[cur]) return;
        for (int i = 0; i < SIGMA; i++) if (~sa[cur][i]) {
            cout << cur << "," << d[cur]-1 << ' ' << sa[cur][i] << 
                "," << d[sa[cur][i]]-1 << ' ' << char(i+'a') << '\n';
            dfs(sa[cur][i], d);
        }
    }
    void prnt() {
        auto d = count_distinct();
        dfs(0, d);
    }
} /* init() insert() */
```