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
struct segtreebeats {
    static const ll INF = 2e18, UNSET = LLONG_MIN;

    struct Node {
        ll sum, mx1, mx2, mxc, mn1, mn2, mnc, d_gcd, lz_add, lz_set;
    };

    int sz;
    vector<Node> tree;

    Node leaf(ll v) { return {v, v, -INF, 1, v, INF, 1, 0, 0, UNSET}; }

    segtreebeats(int n, const vector<ll> &a) {
        for (sz = 1; sz < n; sz <<= 1);
        tree.assign(sz << 1, leaf(0));
        build(a, n, 0, 0, sz - 1);
    }

    Node merge(const Node &L, const Node &R) {
        Node U = {L.sum + R.sum, 0, 0, 0, 0, 0, 0, std::gcd(L.d_gcd, R.d_gcd), 0, UNSET};

        if (L.mx1 == R.mx1) U.mx1 = L.mx1, U.mx2 = max(L.mx2, R.mx2), U.mxc = L.mxc + R.mxc;
        else if (L.mx1 > R.mx1) U.mx1 = L.mx1, U.mx2 = max(L.mx2, R.mx1), U.mxc = L.mxc;
        else U.mx1 = R.mx1, U.mx2 = max(L.mx1, R.mx2), U.mxc = R.mxc;

        if (L.mn1 == R.mn1) U.mn1 = L.mn1, U.mn2 = min(L.mn2, R.mn2), U.mnc = L.mnc + R.mnc;
        else if (L.mn1 < R.mn1) U.mn1 = L.mn1, U.mn2 = min(L.mn2, R.mn1), U.mnc = L.mnc;
        else U.mn1 = R.mn1, U.mn2 = min(L.mn1, R.mn2), U.mnc = R.mnc;

        ll aL = L.mx2, aR = R.mx2;
        if (aL != -INF && aL != L.mn1 && aR != -INF && aR != R.mn1)
            U.d_gcd = std::gcd(U.d_gcd, abs(aL - aR));

        ll any = UNSET;
        if (aL != -INF && aL != L.mn1) any = aL;
        else if (aR != -INF && aR != R.mn1) any = aR;

        for (ll val : {L.mn1, L.mx1, R.mn1, R.mx1}) {
            if (val != U.mn1 && val != U.mx1) {
                if (any != UNSET) U.d_gcd = std::gcd(U.d_gcd, abs(val - any));
                else any = val;
            }
        }
        return U;
    }

    void apply_set(int x, int lx, int rx, ll v) {
        ll len = rx - lx + 1;
        tree[x] = {len * v, v, -INF, len, v, INF, len, 0, 0, v};
    }

    void apply_add(int x, int lx, int rx, ll v) {
        if (!v) return;
        auto &nd = tree[x];
        if (nd.lz_set != UNSET) return apply_set(x, lx, rx, nd.lz_set + v);
        if (nd.mx1 == nd.mn1) return apply_set(x, lx, rx, nd.mn1 + v);
        ll len = rx - lx + 1;
        nd.sum += len * v;
        nd.mx1 += v; if (nd.mx2 != -INF) nd.mx2 += v;
        nd.mn1 += v; if (nd.mn2 != INF) nd.mn2 += v;
        nd.lz_add += v;
    }

    void apply_chmin(int x, int lx, int rx, ll v) {
        auto &nd = tree[x];
        if (nd.mx1 <= v) return;
        if (nd.mn1 >= v) return apply_set(x, lx, rx, v);
        if (nd.mn2 == nd.mx1) nd.mn2 = v;
        nd.sum -= (nd.mx1 - v) * nd.mxc;
        nd.mx1 = v;
    }

    void apply_chmax(int x, int lx, int rx, ll v) {
        auto &nd = tree[x];
        if (nd.mn1 >= v) return;
        if (nd.mx1 <= v) return apply_set(x, lx, rx, v);
        if (nd.mx2 == nd.mn1) nd.mx2 = v;
        nd.sum += (v - nd.mn1) * nd.mnc;
        nd.mn1 = v;
    }

    void push(int x, int lx, int rx) {
        if (lx == rx) return void(tree[x].lz_add = 0, tree[x].lz_set = UNSET);
        int m = (lx + rx) >> 1, lc = x * 2 + 1, rc = x * 2 + 2;

        if (tree[x].lz_set != UNSET) {
            apply_set(lc, lx, m, tree[x].lz_set);
            apply_set(rc, m + 1, rx, tree[x].lz_set);
            tree[x].lz_set = UNSET;
        }
        if (tree[x].lz_add) {
            apply_add(lc, lx, m, tree[x].lz_add);
            apply_add(rc, m + 1, rx, tree[x].lz_add);
            tree[x].lz_add = 0;
        }
        if (tree[lc].mx1 > tree[x].mx1) apply_chmin(lc, lx, m, tree[x].mx1);
        if (tree[rc].mx1 > tree[x].mx1) apply_chmin(rc, m + 1, rx, tree[x].mx1);
        if (tree[lc].mn1 < tree[x].mn1) apply_chmax(lc, lx, m, tree[x].mn1);
        if (tree[rc].mn1 < tree[x].mn1) apply_chmax(rc, m + 1, rx, tree[x].mn1);
    }

    void build(const vector<ll> &a, int n, int x, int lx, int rx) {
        if (lx == rx) return void(tree[x] = (lx < n) ? leaf(a[lx]) : leaf(0));
        int m = (lx + rx) >> 1;
        build(a, n, x * 2 + 1, lx, m);
        build(a, n, x * 2 + 2, m + 1, rx);
        tree[x] = merge(tree[x * 2 + 1], tree[x * 2 + 2]);
    }

    void chmin(int l, int r, ll v, int x, int lx, int rx) {
        if (lx > r || rx < l || tree[x].mx1 <= v) return;
        if (lx >= l && rx <= r && tree[x].mx2 < v) return apply_chmin(x, lx, rx, v);
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        chmin(l, r, v, x * 2 + 1, lx, m);
        chmin(l, r, v, x * 2 + 2, m + 1, rx);
        tree[x] = merge(tree[x * 2 + 1], tree[x * 2 + 2]);
    }

    void chmax(int l, int r, ll v, int x, int lx, int rx) {
        if (lx > r || rx < l || tree[x].mn1 >= v) return;
        if (lx >= l && rx <= r && tree[x].mn2 > v) return apply_chmax(x, lx, rx, v);
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        chmax(l, r, v, x * 2 + 1, lx, m);
        chmax(l, r, v, x * 2 + 2, m + 1, rx);
        tree[x] = merge(tree[x * 2 + 1], tree[x * 2 + 2]);
    }

    void assign(int l, int r, ll v, int x, int lx, int rx) {
        if (lx > r || rx < l) return;
        if (lx >= l && rx <= r) return apply_set(x, lx, rx, v);
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        assign(l, r, v, x * 2 + 1, lx, m);
        assign(l, r, v, x * 2 + 2, m + 1, rx);
        tree[x] = merge(tree[x * 2 + 1], tree[x * 2 + 2]);
    }

    void add(int l, int r, ll v, int x, int lx, int rx) {
        if (lx > r || rx < l) return;
        if (lx >= l && rx <= r) return apply_add(x, lx, rx, v);
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        add(l, r, v, x * 2 + 1, lx, m);
        add(l, r, v, x * 2 + 2, m + 1, rx);
        tree[x] = merge(tree[x * 2 + 1], tree[x * 2 + 2]);
    }

    ll sum(int l, int r, int x, int lx, int rx) {
        if (lx > r || rx < l) return 0;
        if (lx >= l && rx <= r) return tree[x].sum;
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        return sum(l, r, x * 2 + 1, lx, m) + sum(l, r, x * 2 + 2, m + 1, rx);
    }

    ll qmin(int l, int r, int x, int lx, int rx) {
        if (lx > r || rx < l) return INF;
        if (lx >= l && rx <= r) return tree[x].mn1;
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        return min(qmin(l, r, x * 2 + 1, lx, m), qmin(l, r, x * 2 + 2, m + 1, rx));
    }

    ll qmax(int l, int r, int x, int lx, int rx) {
        if (lx > r || rx < l) return -INF;
        if (lx >= l && rx <= r) return tree[x].mx1;
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        return max(qmax(l, r, x * 2 + 1, lx, m), qmax(l, r, x * 2 + 2, m + 1, rx));
    }

    ll qgcd(int l, int r, int x, int lx, int rx) {
        if (lx > r || rx < l) return 0;
        if (lx >= l && rx <= r) {
            ll ans = std::gcd(tree[x].d_gcd, abs(tree[x].mx1));
            if (tree[x].mx2 != -INF) ans = std::gcd(ans, abs(tree[x].mx2 - tree[x].mx1));
            if (tree[x].mn2 != INF) ans = std::gcd(ans, abs(tree[x].mn2 - tree[x].mn1));
            return ans;
        }
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        return std::gcd(qgcd(l, r, x * 2 + 1, lx, m), qgcd(l, r, x * 2 + 2, m + 1, rx));
    }

    // Public Wrappers
    void chmin(int l, int r, ll v) { chmin(l, r, v, 0, 0, sz - 1); }
    void chmax(int l, int r, ll v) { chmax(l, r, v, 0, 0, sz - 1); }
    void assign(int l, int r, ll v) { assign(l, r, v, 0, 0, sz - 1); }
    void add(int l, int r, ll v) { add(l, r, v, 0, 0, sz - 1); }
    ll sum(int l, int r) { return sum(l, r, 0, 0, sz - 1); }
    ll qmin(int l, int r) { return qmin(l, r, 0, 0, sz - 1); }
    ll qmax(int l, int r) { return qmax(l, r, 0, 0, sz - 1); }
    ll qgcd(int l, int r) { return qgcd(l, r, 0, 0, sz - 1); }
};

void solve() {

}