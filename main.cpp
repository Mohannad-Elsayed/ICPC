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
template <int M> struct ModInt {
    template <class T> T binpow(T a, ll b) {
        T res = 1;
        while (b) { if (b & 1) res *= a; a *= a, b >>= 1; }
        return res;
    }
    int v;
    ModInt() : v(0) {}
    ModInt(ll v_) {
        v = v_ % M;
        if (v < 0) v += M;
    }

    bool operator==(ModInt o) const { return v == o.v; };
    bool operator!=(ModInt o) const { return !(*this == o); };

    // ModInt add(ModInt a, ModInt b) { return ((a.v + b.v % M)+M)%M;}
    // ModInt sub(ModInt a, ModInt b) { return add(a, -b); }
    // ModInt mul(ModInt a, ModInt b) { return 1ll * a.v * b.v % M; }
    // ModInt div(ModInt a, ModInt b) { return mul(a, binpow(b, M-2)); }

    ModInt &operator+=(ModInt o) { v = (v + o.v) % M; return *this; }
    ModInt &operator-=(ModInt o) { v = (v - o.v + M) % M; return *this; }
    ModInt &operator*=(ModInt o) { v = 1ll * v * o.v % M; return *this; }
    ModInt &operator/=(ModInt o) { return (*this *= binpow(o, M - 2)); }

    friend ModInt operator+(ModInt a, ModInt b) { return a += b; }
    friend ModInt operator-(ModInt a, ModInt b) { return a -= b; }
    friend ModInt operator*(ModInt a, ModInt b) { return a *= b; }
    friend ModInt operator/(ModInt a, ModInt b) { return a /= b; }
    ModInt operator-() { return 0 - *this; }

    friend istream &operator>>(istream &is, ModInt &a) {
        ll x; is >> x;
        a = ModInt(x);
        return is;
    }
    friend ostream &operator<<(ostream &os, ModInt a) { return os << a.v; }
    friend string to_string(ModInt a) { return to_string(a.v); }
};
using mint = ModInt<998244353>;

void solve() {
    int n; cin >> n;
    vector<mint> a(n), b(n);
    for (auto &i : a) cin >> i;
    for (auto &i : b) cin >> i;
    vector<vector<int>> tree(n);
    for (int i = 1, u, v; i < n; i++) {
        cin >> u >> v; u--, v--;
        tree[u].push_back(v);
        tree[v].push_back(u);
    }

    vector<mint> pfx;
    mint sum = 0;
    function<void(int, int)> dfs = [&](int u, int p) {
        pfx.push_back(u);
        bool leaf = true;
        for (int v : tree[u]) if (v ^ p) dfs(v, u), leaf = false;
        if (leaf) {
            mint c = 0;
            print(pfx);
            for (int i = 0, j = (int)pfx.size() - 1; i < pfx.size(); i++, j--) {
                c += b[pfx[i].v];
                sum += c * a[j];
                print(j, c, sum);
            }
        }
        pfx.pop_back();
    };
    dfs(0, -1);
    print(sum);
}