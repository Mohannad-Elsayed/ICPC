---
title: 2. Data Structures
---
# Mo struct
```cpp {.numberLines}
const int N = 1e5 + 5, sq = 317;
struct query{
    int l, r, i;
    bool operator<(const query &other) const {
        if (l / sq != other.l / sq)
            return l / sq < other.l / sq;
        return (l / sq & 1 ? r < other.r : other.r < r);
    }
};
```
# Mo with updates
```cpp {.numberLines}
const int N = 2e5 + 5;
int a[N];
vector<int> coords;
long long curAns, ans[N];

struct Query {
    int l, r, t, id, blk_l, blk_r;

    bool operator<(const Query& o) const {
        if (blk_l != o.blk_l) return blk_l < o.blk_l;
        if (blk_r != o.blk_r) return (blk_l & 1) ? blk_r < o.blk_r : blk_r > o.blk_r;
        return (blk_r & 1) ? t < o.t : t > o.t;
    }
};

struct Update {
    int p, v;
};

vector<Query> queries;
vector<Update> updates;
int vals[N];

inline void add(int i) {
    if (++vals[a[i]] == 1) curAns += coords[a[i]];
}

inline void del(int i) {
    if (!--vals[a[i]]) curAns -= coords[a[i]];
}

inline void do_update(int t, int l, int r) {
    auto &u = updates[t];
    if (l <= u.p && u.p <= r) del(u.p);
    swap(a[u.p], u.v);
    if (l <= u.p && u.p <= r) add(u.p);
}

void process(int n) {
    constexpr int block = 700;
    for (auto &q: queries) {
        q.blk_l = q.l / block;
        q.blk_r = q.r / block;
    }
    sort(queries.begin(), queries.end());

    int l = 1, r = 0, t = 0;
    for (const auto &q: queries) {
        while (t < q.t) do_update(t++, l, r);
        while (t > q.t) do_update(--t, l, r);
        while (l > q.l) add(--l);
        while (r < q.r) add(++r);
        while (l < q.l) del(l++);
        while (r > q.r) del(r--);

        ans[q.id] = curAns;
    }
}

void solve() {
    int n;
    cin >> n;
    coords.reserve(n << 1);
    for (int i = 0; i < n; i++) cin >> a[i], coords.push_back(a[i]);
    int q, j = 0;
    cin >> q;
    for (int i = 0; i < q; i++) {
        string s;
        cin >> s;
        if (s.front() == 'Q') {
            int l, r;
            cin >> l >> r;
            queries.push_back({--l, --r, (int) updates.size(), j, 0, 0});
            j++;
        } else {
            int idx, val;
            cin >> idx >> val;
            --idx;
            updates.push_back({idx, val});
            coords.push_back(val);
        }
    }

    sort(coords.begin(), coords.end());
    coords.erase(unique(coords.begin(), coords.end()), coords.end());
    for (int i = 0; i < n; i++) 
        a[i] = lower_bound(coords.begin(), coords.end(), a[i]) - coords.begin();
    for (auto &o: updates) 
        o.v = lower_bound(coords.begin(), coords.end(), o.v) - coords.begin();

    // pass n to the process function
    process(n);
    
    for (int i = 0; i < j; i++) cout << ans[i] << ' ';

    queries.clear();
    updates.clear();
    coords.clear();
    curAns = 0;
}
```
# 2D Prefix Sum
```cpp {.numberLines}
// construction
for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
        prefix[i][j] = 
            arr[i][j] 
            + prefix[i - 1][j] 
            + prefix[i][j - 1] 
            - prefix[i - 1][j - 1];
    }
}

// query
pfx[to_row][to_col] 
- pfx[from_row - 1][to_col] 
- pfx[to_row][from_col - 1] 
+ pfx[from_row - 1][from_col - 1];

// partial: add value v to subrectangle [from_row, from_col] to [to_row, to_col]
diff[from_row][from_col]     += v;
diff[from_row][to_col + 1] -= v;
diff[to_row + 1][from_col] -= v;
diff[to_row + 1][to_col + 1] += v;
// then construct diff, grid[i][j] += diff[i][j];
```
# Multiset Lazy deletion
```cpp {.numberLines}
template<typename T, typename Compare = less<T>>
struct MS {
    priority_queue<T, vector<T>, Compare> pq, del;
    void normalize(){
        while(!pq.empty() && !del.empty() && pq.top() == del.top()){
            pq.pop();
            del.pop();
        }
    }
    bool empty() { normalize(); return size() == 0; }
    int size() { return (int)pq.size() - (int)del.size(); }
    void insert(T x) { pq.push(x); }
    void erase(T x) { del.push(x);}
    T top() { normalize(); return pq.top(); }
    void pop() { normalize(); pq.pop(); }
    void clear() {
        while(!pq.empty()) pq.pop();
        while(!del.empty()) del.pop();
    }
};
```
# BIT, Fenwick Tree
```cpp {.numberLines}
struct BIT { // 0-based
    int n;
    vector<ll> tree;
    
    BIT(int size) : n(size + 2), tree(n + 1) {}
    void add(int i, ll val) {
        for(i++; i <= n; i += i & -i) tree[i] += val;
    }
    ll query(int i) {
        ll sum = 0;
        for (i++; i > 0; i &= i - 1) sum += tree[i];
        return sum;
    }
    int lower_bound(ll target) {
        int i = 0;
        ll curr = 0;
        for (int mask = 1 << __lg(n); mask > 0; mask >>= 1) {
            if (i + mask <= n && curr + tree[i + mask] < target) {
                curr += tree[i += mask];
            }
        }
        return i;
    }
};
```
# BIT Range / Fenwick Range
```cpp {.numberLines}
template<typename T>
class BitR { // 0-based
    int n;
    vector<T> f, s;
    void add(vector<T> &a, int i, T val) {
        for(; i < n; i += i & -i)
            a[i] += val;
    }
public:
    BitR(int n) : n(n + 5), f(n + 6), s(n + 6) { }

    void add(int i, T val) { add(s, i + 1, -val); }

    T query_point(int i) {
        if (!i) return query(0);
        return query(i) - query(i - 1);
    }
    void set(int i, T val) {
        int c = query_point(i);
        add(i, val - c);
    }

    void add(int l, int r, T val) {
        l++, r++;
        add(f, l, val);
        add(f, r + 1, -val);
        add(s, l, val * (l - 1));
        add(s, r + 1, -val * r);
    }

    T query(int ii) {
        ii++;
        T sum = 0;
        int i = ii;
        for(; i > 0; i ^= i & -i)
            sum += f[i];
        sum *= ii;
        i = ii;
        for(; i > 0; i ^= i & -i)
            sum -= s[i];
        return sum;
    }

    T range(int l, int r) { return query(r) - query(l - 1); }
};
```
# 2D BIT / 2D fenwick
```cpp {.numberLines}
template<typename T>
struct BIT2D { // 1-based
    int n, m;
    vector<vector<T>> tree;

    BIT2D(int n, int m) : n(n), m(m), tree(n + 2, vector<T>(m + 2, 0)) {}

    void update(int x, int y, T val) {
        for (int i = x; i <= n; i += i & -i) {
            for (int j = y; j <= m; j += j & -j) {
                tree[i][j] += val;
            }
        }
    }

    T getPrefix(int x, int y) {
        if (x <= 0 || y <= 0) return 0;
        T ret = 0; // change default value
        for (int i = x; i > 0; i &= i - 1) {
            for (int j = y; j > 0; j &= j - 1) {
                ret += tree[i][j];
            }
        }
        return ret;
    }

    T getSquare(int xl, int yl, int xr, int yr) { // change operation
        return getPrefix(xr, yr) + getPrefix(xl - 1, yl - 1) - 
               getPrefix(xr, yl - 1) - getPrefix(xl - 1, yr);
    }
};
```
# Segment Tree
```cpp {.numberLines}
struct info {
    long long sum = 0;
    info(long long sum = 0) : sum(sum) {}
    friend info operator+(const info &l, const info &r) {
        return {l.sum + r.sum};
    }
};

template<class info>
struct segmentTreeIterative {
    int n;
    vector<info> tree;

    explicit segmentTreeIterative(int n) : n(n), tree(n << 1) {}

    template<class U>
    explicit segmentTreeIterative(const vector<U> &arr) : n(arr.size()), 
                tree(n << 1) {
        for(int i = 0; i < n; i++) tree[i + n] = info(arr[i]);
        for(int i = n - 1; i > 0; i--) tree[i] = tree[i << 1] + tree[i << 1 | 1];
    }

    void set(int i, info v) {
        for(tree[i += n] = v; i >>= 1; )
            tree[i] = tree[i << 1] + tree[i << 1 | 1];
    }

    info get(int l, int r) {
        info resL, resR;
        for(l += n, r += n + 1; l < r; l >>= 1, r >>= 1) {
            if(l & 1) resL = resL + tree[l++];
            if(r & 1) resR = tree[--r] + resR;
        }
        return resL + resR;
    }
};
```
# Segment Tree (Recursive)
```cpp {.numberLines}
struct info {
    long long sum = 0;
    info(long long sum = 0) : sum(sum) {}
    friend info operator+(const info &l, const info &r) {
        return {l.sum + r.sum};
    }
};

template<class info>
struct dynamicSegmentTree {
    struct node { int l = 0, r = 0; info v; };
    vector<node> tr;
    int n, root = 0;

    explicit dynamicSegmentTree(int n = 1e9, int expectedOps = 1) : n(n), tr(1) { 
        tr.reserve(expectedOps * __lg(n) * 2); 
    }

    info get(int x, int lx, int rx, int l, int r) {
        if (!x || lx > r || l > rx) return info();
        if (lx >= l && rx <= r) return tr[x].v;
        int m = (lx + rx) >> 1;
        return get(tr[x].l, lx, m, l, r) + get(tr[x].r, m + 1, rx, l, r);
    }

    int set(int x, int lx, int rx, int i, info val) {
        if (i < lx || i > rx) return x;
        if (!x) x = tr.size(), tr.emplace_back();
        if (lx == rx) return tr[x].v = val, x;
        int m = (lx + rx) >> 1;
        tr[x].l = set(tr[x].l, lx, m, i, val);
        tr[x].r = set(tr[x].r, m + 1, rx, i, val);
        return tr[x].v = tr[tr[x].l].v + tr[tr[x].r].v, x;
    }

    void set(int i, info v) { root = set(root, 0, n, i, v); }
    info get(int l, int r) { return get(root, 0, n, l, r); }
};
```
# 2D Segment Tree
```cpp {.numberLines}
struct info {
    long long sum = 0;
    info(long long sum = 0) : sum(sum) {}
    friend info operator+(const info &l, const info &r) {
        return {l.sum + r.sum};
    }
};

template<class info>
struct segmentTree2d {
    int nx, ny;
    vector<vector<info>> tree;

    explicit segmentTree2d(int n, int m) : nx(n), ny(m), tree(nx << 1, 
        vector<info>(ny << 1)) {}

    template<class U>
    explicit segmentTree2d(const vector<vector<U>> &a) {
        nx = a.size();
        ny = nx ? a[0].size() : 0;
        tree.assign(nx << 1, vector<info>(ny << 1));

        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) tree[i + nx][j + ny] = info(a[i][j]);
            for(int y = ny - 1; y > 0; y--)
                tree[i + nx][y] = tree[i + nx][y << 1] + tree[i + nx][y << 1 | 1];
        }
        for(int x = nx - 1; x > 0; x--)
            for(int y = 0; y < (ny << 1); y++)
                tree[x][y] = tree[x << 1][y] + tree[x << 1 | 1][y];
    }

    void set(int i, int j, info v) {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return;
        int x = i + nx, y = j + ny;
        tree[x][y] = v;
        for(int yy = y >> 1; yy > 0; yy >>= 1)
            tree[x][yy] = tree[x][yy << 1] + tree[x][yy << 1 | 1];

        for(int xx = x >> 1; xx > 0; xx >>= 1) {
            tree[xx][y] = tree[xx << 1][y] + tree[xx << 1 | 1][y];
            for(int yy = y >> 1; yy > 0; yy >>= 1)
                tree[xx][yy] = tree[xx][yy << 1] + tree[xx][yy << 1 | 1];
        }
    }

    info queryY(int nodeX, int y1, int y2) const {
        info resL, resR;
        for(int l = y1 + ny, r = y2 + ny + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) resL = resL + tree[nodeX][l++];
            if (r & 1) resR = tree[nodeX][--r] + resR;
        }
        return resL + resR;
    }

    info get(int x1, int y1, int x2, int y2) {
        if (!nx || !ny) return info();
        x1 = max(x1, 0); y1 = max(y1, 0);
        x2 = min(x2, nx - 1); y2 = min(y2, ny - 1);
        if (x1 > x2 || y1 > y2) return info();

        info resL, resR;
        for(int l = x1 + nx, r = x2 + nx + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) resL = resL + queryY(l++, y1, y2);
            if (r & 1) resR = queryY(--r, y1, y2) + resR;
        }
        return resL + resR;
    }
};
```
# Lazy Segment Tree
```cpp {.numberLines}
struct tag {
    ll add = 0;
    void apply(const tag &p) {
        add += p.add;
    }
};

struct info {
    ll sum = 0;
    info(ll sum = 0) : sum(sum) {}
    void apply(const tag &lz, int lx, int rx) {
        sum += 1ll * (rx - lx + 1) * lz.add;
    }
    friend info operator+(const info &l, const info &r) {
        return {l.sum + r.sum};
    }
};

template<class info, class tag>
struct lazySegment {
    int n;
    vector<info> tr;
    vector<tag> lz;

    explicit lazySegment(int _n) : n(_n + 1), tr(4 << __lg(n)), lz(4 << __lg(n)) {}
    template<class U>
    explicit lazySegment(const U &arr) : n(arr.size()), tr(4 << __lg(n)), 
                lz(4 << __lg(n)) {
        build(1, 0, n - 1, arr);
    }

    void push(int x, int lx, int rx) {
        if (lx != rx) lz[x << 1].apply(lz[x]), lz[x << 1 | 1].apply(lz[x]);
        tr[x].apply(lz[x], lx, rx);
        lz[x] = tag();
    }

    template<class U>
    void build(int x, int lx, int rx, const U &arr) {
        if (lx == rx) return void(tr[x] = arr[lx]);
        int m = (lx + rx) >> 1;
        build(x << 1, lx, m, arr), build(x << 1 | 1, m + 1, rx, arr);
        tr[x] = tr[x << 1] + tr[x << 1 | 1];
    }

    info get(int x, int lx, int rx, int l, int r) {
        push(x, lx, rx);
        if (lx > r || l > rx) return info();
        if (lx >= l && rx <= r) return tr[x];
        int m = (lx + rx) >> 1;
        return get(x << 1, lx, m, l, r) + get(x << 1 | 1, m + 1, rx, l, r);
    }

    void set(int x, int lx, int rx, int i, info val) {
        push(x, lx, rx);
        if (i < lx || i > rx) return;
        if (lx == rx) return void(tr[x] = val);
        int m = (lx + rx) >> 1;
        set(x << 1, lx, m, i, val), set(x << 1 | 1, m + 1, rx, i, val);
        tr[x] = tr[x << 1] + tr[x << 1 | 1];
    }

    void applyRange(int x, int lx, int rx, int l, int r, tag v) {
        push(x, lx, rx);
        if (lx > r || l > rx) return;
        if (lx >= l && rx <= r) return lz[x] = v, push(x, lx, rx);
        int m = (lx + rx) >> 1;
        applyRange(x << 1, lx, m, l, r, v), 
        applyRange(x << 1 | 1, m + 1, rx, l, r, v);
        tr[x] = tr[x << 1] + tr[x << 1 | 1];
    }

    void set(int i, info v) { set(1, 0, n - 1, i, v); }
    void applyRange(int l, int r, tag v) { applyRange(1, 0, n - 1, l, r, v); }
    info get(int l, int r) { return get(1, 0, n - 1, l, r); }
};
```
# Dynamic Lazy Segment Tree
```cpp {.numberLines}
template<class info, class tag>
struct dynamicLazySegmentTree {
    struct node { int l = 0, r = 0; info v; tag t; };
    vector<node> tr;
    int n, root = 0;

    explicit dynamicLazySegmentTree(int n = 1e9) : n(n), tr(1) {}

    void push(int x, int lx, int rx) {
        if (lx == rx) return void(tr[x].t = tag());
        if (!tr[x].l) tr[x].l = tr.size(), tr.emplace_back();
        if (!tr[x].r) tr[x].r = tr.size(), tr.emplace_back();

        int m = (lx + rx) >> 1, l = tr[x].l, r = tr[x].r;
        tr[l].t.apply(tr[x].t), tr[l].v.apply(tr[x].t, lx, m);
        tr[r].t.apply(tr[x].t), tr[r].v.apply(tr[x].t, m + 1, rx);
        tr[x].t = tag();
    }

    int set(int x, int lx, int rx, int i, info val) {
        if (i < lx || i > rx) return x;
        if (!x) x = tr.size(), tr.emplace_back();
        if (lx == rx) {
            tr[x].v = val, tr[x].t = tag();
            return x;
        }
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        tr[x].l = set(tr[x].l, lx, m, i, val);
        tr[x].r = set(tr[x].r, m + 1, rx, i, val);
        return tr[x].v = tr[tr[x].l].v + tr[tr[x].r].v, x;
    }

    int applyRange(int x, int lx, int rx, int l, int r, tag val) {
        if (lx > r || l > rx) return x;
        if (!x) x = tr.size(), tr.emplace_back();
        if (lx >= l && rx <= r) {
            tr[x].t.apply(val);
            tr[x].v.apply(val, lx, rx);
            return x;
        }
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        tr[x].l = applyRange(tr[x].l, lx, m, l, r, val);
        tr[x].r = applyRange(tr[x].r, m + 1, rx, l, r, val);
        return tr[x].v = tr[tr[x].l].v + tr[tr[x].r].v, x;
    }

    info get(int x, int lx, int rx, int l, int r) {
        if (!x || lx > r || l > rx) return info();
        if (lx >= l && rx <= r) return tr[x].v;
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        return get(tr[x].l, lx, m, l, r) + get(tr[x].r, m + 1, rx, l, r);
    }

    void set(int i, info v) { root = set(root, 0, n, i, v); }
    void applyRange(int l, int r, tag v) { root = applyRange(root, 0, n, l, r, v); }
    info get(int l, int r) { return get(root, 0, n, l, r); }
};
```
# Segment Tree Beats
```cpp {.numberLines}
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
        Node U = {L.sum + R.sum, 0, 0, 0, 0, 0, 0, gcd(L.d_gcd, R.d_gcd), 0, UNSET};
        
if (L.mx1 == R.mx1) U.mx1 = L.mx1, U.mx2 = max(L.mx2, R.mx2), U.mxc = L.mxc + R.mxc;
else if (L.mx1 > R.mx1) U.mx1 = L.mx1, U.mx2 = max(L.mx2, R.mx1), U.mxc = L.mxc;
else U.mx1 = R.mx1, U.mx2 = max(L.mx1, R.mx2), U.mxc = R.mxc;

if (L.mn1 == R.mn1) U.mn1 = L.mn1, U.mn2 = min(L.mn2, R.mn2), U.mnc = L.mnc + R.mnc;
else if (L.mn1 < R.mn1) U.mn1 = L.mn1, U.mn2 = min(L.mn2, R.mn1), U.mnc = L.mnc;
else U.mn1 = R.mn1, U.mn2 = min(L.mn1, R.mn2), U.mnc = R.mnc;

        ll aL = L.mx2, aR = R.mx2;
        if (aL != -INF && aL != L.mn1 && aR != -INF && aR != R.mn1)
            U.d_gcd = gcd(U.d_gcd, abs(aL - aR));

        ll any = UNSET;
        if (aL != -INF && aL != L.mn1) any = aL;
        else if (aR != -INF && aR != R.mn1) any = aR;

        for (ll val : {L.mn1, L.mx1, R.mn1, R.mx1}) {
            if (val != U.mn1 && val != U.mx1) {
                if (any != UNSET) U.d_gcd = gcd(U.d_gcd, abs(val - any));
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
        if (lx == rx) return tree[x].lz_add = 0, tree[x].lz_set = UNSET, void();
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
            ll ans = gcd(tree[x].d_gcd, abs(tree[x].mx1));
            if (tree[x].mx2 != -INF) ans = gcd(ans, abs(tree[x].mx2 - tree[x].mx1));
            if (tree[x].mn2 != INF) ans = gcd(ans, abs(tree[x].mn2 - tree[x].mn1));
            return ans;
        }
        push(x, lx, rx);
        int m = (lx + rx) >> 1;
        return gcd(qgcd(l, r, x * 2 + 1, lx, m), qgcd(l, r, x * 2 + 2, m + 1, rx));
    }

    // Public Wrappers
    void chmin(int l, int r, ll v) { chmin(l, r, v, 0, 0, sz - 1); }
    void chmax(int l, int r, ll v) { chmax(l, r, v, 0, 0, sz - 1); }
    void assign(int l, int r, ll v) { assign(l, r, v, 0, 0, sz - 1); }
    void add(int l, int r, ll v) { add(l, r, v, 0, 0, sz - 1); }
    ll qsum(int l, int r) { return sum(l, r, 0, 0, sz - 1); }
    ll qmin(int l, int r) { return qmin(l, r, 0, 0, sz - 1); }
    ll qmax(int l, int r) { return qmax(l, r, 0, 0, sz - 1); }
    ll qgcd(int l, int r) { return qgcd(l, r, 0, 0, sz - 1); }
};
```
# Dynamic Persistent Segment Tree
```cpp {.numberLines}
template<class info>
struct DPSGT {
    struct node { int l = 0, r = 0; info v; };
    vector<node> tr;
    int n;

    explicit DPSGT(int n = 1e9, int expectedOps = 1) : n(n), tr(1) { 
        tr.reserve(expectedOps * __lg(n) * 2); 
    }

    info get(int x, int lx, int rx, int l, int r) {
        if (!x || lx > r || l > rx) return info();
        if (lx >= l && rx <= r) return tr[x].v;
        int m = (lx + rx) >> 1;
        return get(tr[x].l, lx, m, l, r) + get(tr[x].r, m + 1, rx, l, r);
    }

    int set(int x, int lx, int rx, int i, info val) {
        if (i < lx || i > rx) return x;
        int nx = tr.size(); 
        tr.push_back(tr[x]);
        if (lx == rx) return tr[nx].v = val, nx;
        int m = (lx + rx) >> 1;
        tr[nx].l = set(tr[nx].l, lx, m, i, val);
        tr[nx].r = set(tr[nx].r, m + 1, rx, i, val);
        return tr[nx].v = tr[tr[nx].l].v + tr[tr[nx].r].v, nx;
    }

    int set(int root, int i, info v) { return set(root, 0, n, i, v); }
    info get(int root, int l, int r) { return get(root, 0, n, l, r); }
};
```
# Wavelet Tree
```cpp {.numberLines}
struct WaveletTree {
    int lo, hi;
    WaveletTree *lc = 0, *rc = 0;
    vector<int> b;

    template<class It>
    WaveletTree(It from, It to, int x, int y) : lo(x), hi(y) {
        if (from >= to || lo == hi) return;
        int mid = lo + (hi - lo) / 2;

        b.reserve(distance(from, to) + 1);
        b.push_back(0);
        for (auto it = from; it != to; ++it) {
            b.push_back(b.back() + (*it <= mid));
        }

        auto pivot = stable_partition(from, to, [mid](int val) 
            { return val <= mid; });
        lc = new WaveletTree(from, pivot, lo, mid);
        rc = new WaveletTree(pivot, to, mid + 1, hi);
    }

    // the passed vector is changed, pass a copy in the main
    // Takes a vector of integers, and the min/max values in it.
    // The vector should be 0-indexed but queries works as 1-based
    WaveletTree(vector<int>& a, int x, int y) :
    WaveletTree(a.begin(), a.end(), x, y) {}

    ~WaveletTree() { delete lc; delete rc; }

    int kth(int l, int r, int k) {
        if (l > r) return 0;
        if (lo == hi) return lo;
        int in_left = b[r] - b[l - 1], lb = b[l - 1], rb = b[r];
        if (k <= in_left) return lc->kth(lb + 1, rb, k);
        return rc->kth(l - lb, r - rb, k - in_left);
    }

    int LTE(int l, int r, int k) {
        if (l > r || k < lo) return 0;
        if (hi <= k) return r - l + 1;
        int lb = b[l - 1], rb = b[r];
        return lc->LTE(lb + 1, rb, k) + rc->LTE(l - lb, r - rb, k);
    }

    int count(int l, int r, int k) {
        if (l > r || k < lo || k > hi) return 0;
        if (lo == hi) return r - l + 1;
        int mid = lo + (hi - lo) / 2, lb = b[l - 1], rb = b[r];
        if (k <= mid) return lc->count(lb + 1, rb, k);
        return rc->count(l - lb, r - rb, k);
    }
};
```
# Implicit Treap
```cpp {.numberLines}
// You must manually track the root of your tree in your main function:
//      Treap tree;
//      int root = 0; 
// 1. Insert val at index i:
//      int l, r;
//      tree.split(root, i, l, r);
//      int mid = tree.new_node(val);
//      root = tree.merge(tree.merge(l, mid), r);
//
// 2. Delete the element at index i:
//      auto [l, mid, r] = tree.split(root, i, i);
//      root = tree.merge(l, r); // mid is safely dropped and ignored
//
// 3. Range Sum for subarray [L, R]:
//      auto [l, mid, r] = tree.split(root, L, R);
//      long long ans = tree.tr[mid].sum;
//      root = tree.merge(tree.merge(l, mid), r); // ALWAYS merge back!
//
// 4. Range Reverse for subarray [L, R]:
//      auto [l, mid, r] = tree.split(root, L, R);
//      tree.tr[mid].rev ^= 1; // apply the lazy tag
//      root = tree.merge(tree.merge(l, mid), r); // ALWAYS merge back!
// ------------------------------------------

mt19937 rnd(time(nullptr));
struct Treap {
    struct node {
        uint32_t pri = rnd();
        int sz = 1, l = 0, r = 0, val{};
        int64_t sum{};
        bool rev = false;
        node(int x) : sum(val = x) { }
    };
    vector<node> tr;
    
    // tr[0] acts as a dummy/null node to avoid segmentation faults
    Treap() : tr(1, 0) { tr[0].sz = 0; }
    
    inline void pull(int x) {
        tr[x].sz = tr[tr[x].l].sz + tr[tr[x].r].sz + 1;
        tr[x].sum = tr[tr[x].l].sum + tr[tr[x].r].sum + tr[x].val;
    }
    
    inline void push(int x) {
        if(tr[x].rev) {
            tr[tr[x].l].rev ^= 1, tr[tr[x].r].rev ^= 1;
            swap(tr[x].l, tr[x].r);
            tr[x].rev = false;
        }
    }
    
    // Returns the index of the newly created node
    inline int new_node(int val) {
        tr.emplace_back(val);
        return int(tr.size()) - 1;
    }
    
    // Concatenates treap rx to the right of lx. Returns the new root.
    int merge(int lx, int rx) {
        if(!lx || !rx) return rx + lx;
        push(lx), push(rx);
        if(tr[lx].pri < tr[rx].pri) 
            return tr[rx].l = merge(lx, tr[rx].l), pull(rx), rx;
        return tr[lx].r = merge(tr[lx].r, rx), pull(lx), lx;
    }
    
    // Splits treap x into lx (first k elements) and rx (the rest).
    void split(int x, int k, int &lx, int &rx) {
        if(!x) return lx = rx = 0, void();
        push(x);
        if(k <= tr[tr[x].l].sz) split(tr[x].l, k, lx, tr[x].l), pull(rx = x);
        else split(tr[x].r, k - tr[tr[x].l].sz - 1, tr[x].r, rx), pull(lx = x);
    }
    
    // Helper to extract a specific subarray [l, r] into the middle partition
    array<int, 3> split(int x, int l, int r) {
        int a, b, c;
        split(x, r + 1, b, c);
        split(b, l, a, b);
        return {a, b, c};
    }
};
```
# Sparse Table
```cpp {.numberLines}
struct ST {
    static const int K = 18, N = 2e5 + 5;
    int st[K][N];

    void build(int n, const vector<int>& a) {
        for(int i = 0; i < n; i++) st[0][i] = a[i];
        for(int k = 1; k < K; k++)
            for(int i = 0; i + (1 << k) <= n; i++)
                st[k][i] = min(st[k - 1][i], st[k - 1][i + (1 << (k - 1))]);
    }

    int query(int l, int r) {
        int k = __lg(r - l + 1);
        return min(st[k][l], st[k][r - (1 << k) + 1]);
    }
};


template<class T, class F>
struct sparse {
    int n, Log;
    vector<vector<T>> table;
    F merge;
    T id;

    sparse(const vector<T>& arr, F merge, T id = T()) :
            n(arr.size()), Log(__lg(n) + 1), table(Log, vector<T>(n)), 
            merge(merge), id(id) {
        table[0] = arr;
        for (int l = 1; l < Log; l++) {
            for (int i = 0; i + (1 << (l - 1)) < n; i++) {
                table[l][i] = merge(table[l - 1][i], 
                table[l - 1][i + (1 << (l - 1))]);
            }
        }
    }

    T query(int l, int r) {
        if (l > r) return id;
        int len = __lg(r - l + 1);
        return merge(table[len][l], table[len][r - (1 << len) + 1]);
    }

    T query_log(int l, int r) {
        T res = id;
        bool first = true;
        for (int j = Log - 1; j >= 0; j--) {
            if (1 << j <= r - l + 1) {
                res = first ? table[j][l] : merge(res, table[j][l]);
                first = false;
                l += 1 << j;
            }
        }
        return res;
    }
};
// sparse st(a, [](int x, int y) { return min(x, y); }, INT_MAX);
```
# 2D Sparse Table
```cpp {.numberLines}
#include <bits/stdc++.h>
using namespace std;

template<class T, class F>
struct sparse2d {
    int n, m, Kx, Ky;
    vector<int> lgx, lgy;
    vector<vector<vector<vector<T>>>> st;
    F merge;

    sparse2d(const vector<vector<T>>& a, F merge) : merge(merge) {
        n = (int)a.size();
        m = (int)a[0].size();

        lgx.resize(n + 1);
        lgy.resize(m + 1);
        for (int i = 2; i <= n; ++i) lgx[i] = lgx[i / 2] + 1;
        for (int j = 2; j <= m; ++j) lgy[j] = lgy[j / 2] + 1;

        Kx = lgx[n];
        Ky = lgy[m];

        st.assign(Kx + 1, vector<vector<vector<T>>>(Ky + 1));

        st[0][0] = a;

        // Build along columns for kx = 0
        for (int ky = 1; ky <= Ky; ++ky) {
            int len = 1 << ky;
            int half = len >> 1;
            st[0][ky].assign(n, vector<T>(m - len + 1));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j + len <= m; ++j) {
                    st[0][ky][i][j] = merge(st[0][ky - 1][i][j],
                                             st[0][ky - 1][i][j + half]);
                }
            }
        }

        // Build along rows for all ky
        for (int kx = 1; kx <= Kx; ++kx) {
            int lenx = 1 << kx;
            int halfx = lenx >> 1;
            for (int ky = 0; ky <= Ky; ++ky) {
                int leny = 1 << ky;
                st[kx][ky].assign(n - lenx + 1, vector<T>(m - leny + 1));
                for (int i = 0; i + lenx <= n; ++i) {
                    for (int j = 0; j + leny <= m; ++j) {
                        st[kx][ky][i][j] = merge(st[kx - 1][ky][i][j],
                                                 st[kx - 1][ky][i + halfx][j]);
                    }
                }
            }
        }
    }

    // inclusive coordinates: [x1..x2], [y1..y2]
    T query(int x1, int y1, int x2, int y2) const {
        int kx = lgx[x2 - x1 + 1];
        int ky = lgy[y2 - y1 + 1];
        int dx = 1 << kx;
        int dy = 1 << ky;

        const T& a = st[kx][ky][x1][y1];
        const T& b = st[kx][ky][x2 - dx + 1][y1];
        const T& c = st[kx][ky][x1][y2 - dy + 1];
        const T& d = st[kx][ky][x2 - dx + 1][y2 - dy + 1];

        return merge(merge(a, b), merge(c, d));
    }
};
```
# DSU
```cpp {.numberLines}
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
```
# Persistent DSU
```cpp {.numberLines}
struct persistent_DSU {
    vector<vector<pair<int, int> > > par;
    persistent_DSU(int n) : par(n + 1, {{-1, 0}}) { }

    int time = 0;
    bool merge(int u, int v) {
        ++time;
        if ((u = root(u, time)) == (v = root(v, time)))
            return 0;
        if (par[u].back().first > par[v].back().first)
            swap(u, v);
        par[u].push_back({par[u].back().first + par[v].back().first, time});
        par[v].push_back({u, time}); // par[v] = u
        return 1;
    }

    bool same(int u, int v, int t) {
        return root(u, t) == root(v, t);
    }

    int root(int u, int t) {
        // root of u at time t
        if (par[u].back().first >= 0 && par[u].back().second <= t)
            return root(par[u].back().first, t);
        return u;
    }

    int size(int u, int t) { // O(log)
        // size of the component of u at time t
        u = root(u, t);
        int l = 0, r = (int) par[u].size() - 1, ans = 0;
        while (l <= r) {
            int mid = l + r >> 1;
            if (par[u][mid].second <= t)
                ans = mid, l = mid + 1;
            else
                r = mid - 1;
        }
        return -par[u][ans].first;
    }
};
```
# Bipartite DSU
```cpp {.numberLines}
// Maintains whether each component is bipartite
struct BipartiteDSU {
    vector<int> sz, bipartite;
    vector<pair<int, int> > par;

    BipartiteDSU(int n) : par(n), sz(n, 1), bipartite(n) {
        for (int i = 0; i < n; ++i) {
            par[i] = {i, 0};
        }
    }

    pair<int, int> find(int u) {
        if (u == par[u].first) return {u, 0};
        int parity = par[u].second;
        par[u] = find(par[u].first);
        par[u].second ^= parity;
        return par[u];
    }

    bool same(int x, int y) {
        return find(x).first == find(y).first;
    }

    bool join(int u, int v) {
        pair<int, int> pu = find(u);
        pair<int, int> pv = find(v);
        u = pu.first;
        v = pv.first;
        int x = pu.second, y = pv.second;
        if (u == v) {
            if (x == y) bipartite[u] = false;
            return false;
        }
        if (sz[u] < sz[v]) swap(u, v);
        par[v] = {u, x ^ y ^ 1};
        bipartite[u] &= bipartite[v];
        sz[u] += sz[v];
        return true;
    }

    int size(int x) { return sz[find(x).first]; }
};
```
# Monotonic Stack / Queue
```cpp {.numberLines}
template<class T>
struct Mono_stack {
    stack<pair<T, T> > st;

    void push(const T &val) {
        if (st.empty()) st.emplace(val, val);
        else st.emplace(val, max(val, st.top().second));
    }

    void pop() { st.pop(); }
    bool empty() { return st.empty(); }
    int size() { return st.size(); }
    T top() { return st.top().first; }
    T max() { return st.top().second; }
};

template<class T>
struct Mono_queue {
    Mono_stack<T> pop_st, push_st;

    void push(const T &val) { push_st.push(val); }

    void move() {
        if (pop_st.size()) return;
        while (!push_st.empty()) pop_st.push(push_st.top()), push_st.pop();
    }

    void pop() { move(); pop_st.pop(); }
    bool empty() { return pop_st.empty() && push_st.empty(); }
    int size() { return pop_st.size() + push_st.size(); }
    T top() { move(); return pop_st.top(); }

    T max() {
        if (pop_st.empty()) return push_st.max();
        if (push_st.empty()) return pop_st.max();
        return max(push_st.max(), pop_st.max());
    }
};
```
# Ordered Data Structures (pb_ds)
```cpp {.numberLines}
#include <ext/pb_ds/assoc_container.hpp> 
#include <ext/pb_ds/tree_policy.hpp> 
using namespace __gnu_pbds;
template <typename T> using ordered_set = tree<T, null_type, less<T>, 
    rb_tree_tag, tree_order_statistics_node_update>;
template <typename T, typename R> using ordered_map = tree<T, R, less<T>, 
    rb_tree_tag, tree_order_statistics_node_update>;
```
# BucketList
```cpp {.numberLines}
template<class T>
struct BucketList {
    int siz = 0;
    vector<vector<T> > a;
    static constexpr int SPLIT_RATIO = 28;

    void insert(int i, T x) {
        if (siz == 0) {
            siz = 1;
            a = {{x}};
            return;
        }
        siz++;
        for (ll bi = 0; bi < size(a); bi++) {
            auto &bucket = a[bi];
            if (i <= size(bucket)) {
                bucket.insert(begin(bucket) + i, x);
                if (size(bucket) > size(a) * SPLIT_RATIO) {
                    auto L = end(bucket) - size(bucket) / 2, R = end(bucket);
                    a.emplace(begin(a) + bi + 1, L, R);
                    // bucket might be broken
                    a[bi].erase(L, R);
                }
                return;
            }
            i -= size(bucket);
        }
    }

    void erase(int i) {
        siz--;
        for (int bi = 0; bi < size(a); bi++) {
            auto& bucket = a[bi];
            if (i < size(bucket)) {
                bucket.erase(begin(bucket) + i);
                if (bucket.empty()) a.erase(begin(a) + bi);
                return;
            }
            i -= size(bucket);
        }
    }

    T& access(int i) {
        for (auto& bucket : a) {
            if (i < size(bucket)) return bucket[i];
            i -= size(bucket);
        }
        assert(false);
    }
};
```
# Full Dynamic Array
```cpp {.numberLines}
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
template<typename T>
struct DynArray {
    struct Node {
        T val; int pri, sz;
        Node *l, *r;
        Node(const T& v) : val(v), pri(rng()), sz(1), l(0), r(0) {}
    };

    Node* root = 0;

    static int sz(Node* t) { return t ? t->sz : 0; }
    static void pull(Node* t) { if (t) t->sz = 1 + sz(t->l) + sz(t->r); }

    static void split(Node* t, int k, Node*& l, Node*& r) {
        if (!t) { l = r = 0; return; }
        if (sz(t->l) < k) {
            split(t->r, k - sz(t->l) - 1, t->r, r);
            l = t;
        } else {
            split(t->l, k, l, t->l);
            r = t;
        }
        pull(t);
    }

    static void merge(Node*& t, Node* l, Node* r) {
        if (!l || !r) { t = l ? l : r; return; }
        if (l->pri > r->pri) {
            merge(l->r, l->r, r);
            t = l;
        } else {
            merge(r->l, l, r->l);
            t = r;
        }
        pull(t);
    }

    void insert(int pos, const T& val) {
        Node *l, *r;
        split(root, pos, l, r);
        Node* nd = new Node(val);
        merge(l, l, nd);
        merge(root, l, r);
    }

    void erase(int pos) {
        Node *l, *m, *r;
        split(root, pos, l, m);
        split(m, 1, m, r);
        delete m;
        merge(root, l, r);
    }

    T& access(int pos) {
        Node* t = root;
        while (true) {
            int left = sz(t->l);
            if (pos < left) t = t->l;
            else if (pos == left) return t->val;
            else { pos -= left + 1; t = t->r; }
        }
    }

    int size() { return sz(root); }
};
```
# Count below
```cpp {.numberLines}
// counts how many pairs that sum <= limit
template <typename T>
ll count_below(vector<T>& v, int sz, ll Limit){ 
    // unique pairs such that v[i]+v[j] <= limit
    assert(is_sorted(v.begin(), v.end()));
    ll total = 0;
    for (int l = 0, r = sz-1; l < r; l++){
        while(r > l && v[l] + v[r] > Limit)
            r--;
        total += max(0, r-l);
    }
    return total;
}
```