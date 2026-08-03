---
title: 7. Bit Manipulation
---
# Binary Trie (s)
```cpp {.numberLines}
// MXN should generally be (Max Queries * MXB)
const int MXN = 500500 * 30, S = 2, MXB = 30;
int nxt[MXN][S], cnt[MXN], nodes = 1;
// clear the arrays you add here in clear() function
void clear() {
    for (int i = 0; i < nodes; i++) {
        nxt[i][0] = nxt[i][1] = 0;
        cnt[i] = 0;
    }
    nodes = 1;
}
void trieinsert(long long val) {
    int u = 0;
    for (int i = MXB - 1; i >= 0; i--) {
        int bit = (val >> i) & 1;
        if (!nxt[u][bit]) nxt[u][bit] = nodes++;
        u = nxt[u][bit];
        cnt[u]++;
    }
}
int triecount(long long val) {
    int u = 0;
    for (int i = MXB - 1; i >= 0; i--) {
        int bit = (val >> i) & 1;
        if (!nxt[u][bit]) return 0;
        u = nxt[u][bit];
    }
    return cnt[u];
}
void trieerase(long long val) {
    if (!triecount(val)) return;
    int u = 0;
    for (int i = MXB - 1; i >= 0; i--) {
        int bit = (val >> i) & 1;
        u = nxt[u][bit];
        cnt[u]--;
    }
}
long long get_max_xor(long long val) {
    if (nodes == 1 || cnt[nxt[0][0]] + cnt[nxt[0][1]] == 0) return -1;
    int u = 0;
    long long ans = 0;
    for (int i = MXB - 1; i >= 0; i--) {
        int bit = (val >> i) & 1;
        int opp = bit ^ 1;
        if (nxt[u][opp] && cnt[nxt[u][opp]] > 0) {
            ans |= (1LL << i);
            u = nxt[u][opp];
        } else {
            u = nxt[u][bit];
        }
    }
    return ans;
}
long long get_min_xor(long long val) {
    if (nodes == 1 || cnt[nxt[0][0]] + cnt[nxt[0][1]] == 0) return -1;
    int u = 0;
    long long ans = 0;
    for (int i = MXB - 1; i >= 0; i--) {
        int bit = (val >> i) & 1;
        if (nxt[u][bit] && cnt[nxt[u][bit]] > 0) {
            u = nxt[u][bit];
        } else {
            ans |= (1LL << i);
            u = nxt[u][bit ^ 1];
        }
    }
    return ans;
}
// Returns the count of numbers 'x' in the trie such that (x ^ val) < k
int query(long long val, long long k) {
    int u = 0, ans = 0;
    for (int i = MXB - 1; i >= 0; i--) {
        if (!u) break;
        int v_bit = (val >> i) & 1;
        int k_bit = (k >> i) & 1;

        if (k_bit == 1) {
            // If k's bit is 1, taking v_bit makes the XOR bit 0, which is strictly less than k.
            // We add all elements in that subtree to our answer.
            if (nxt[u][v_bit]) ans += cnt[nxt[u][v_bit]];
            // Then we traverse down the opposite branch to evaluate lower bits.
            u = nxt[u][v_bit ^ 1];
        } else {
            // If k's bit is 0, we MUST take v_bit to keep the XOR bit 0.
            u = nxt[u][v_bit];
        }
    }
    return ans;
}
// clear() insert(), remove() query()
```
# Xor Basis
```cpp {.numberLines}
template<const int Log = 30, typename T = ll>
struct basis {
    int sz = 0;
    array<T, Log> a{};
    void add(T x) {
        if(sz == Log) return;
        int i;
        while(x) {
            if(!a[i = __lg(x)]) return sz++, void(a[i] = x);
            x ^= a[i];
        }
    }
    T reduce(T x) {
        if(sz == Log) return 0;
        T res = 0;
        int i;
        while(x) {
            if(a[i = __lg(x)]) x ^= a[i];
            else res |= T(1) << i, x ^= T(1) << i;
        }
        return res;
    }
    bool find(T x) {
        if(sz == Log) return true;
        int i;
        while(x) {
            if(a[i = __lg(x)]) x ^= a[i];
            else return false;
        }
        return true;
    }
    void clear() {
        if(sz) a.fill(0), sz = 0;
    }
    T getMax(T r = 0) {
        for(int i = Log - 1; i >= 0; i--) r = max(r ^ a[i], r);
        return r;
    }

    T find_k(size_t k, T base_val = 0) {
        assert(k < 1ULL << sz);
        T curr = base_val;
        for(int i = Log - 1, b = sz - 1; i >= 0; i--) {
            if(a[i]) {
                if((k >> b & 1) ^ (curr >> i & 1)) curr ^= a[i];
                b--;
            }
        }
        return curr;
    }

    T getMaxBounded(T limit, T r = 0) {
        T best = -1;
        for (int i = Log - 1; i >= 0; i--) {
            if (a[i]) {
                T r1 = r, r2 = r ^ a[i];
                if ((r1 >> i) & 1) swap(r1, r2);

                if ((limit >> i) & 1) {
                    T temp = r1;
                    for (int j = i - 1; j >= 0; j--) 
                        temp = max(temp ^ a[j], temp);
                    best = max(best, temp);
                    r = r2;
                } else 
                    r = r1;
            } else {
                int bit_r = (r >> i) & 1;
                int bit_l = (limit >> i) & 1;

                if (bit_l == 1 && bit_r == 0) {
                    T temp = r;
                    for (int j = i - 1; j >= 0; j--) 
                        temp = max(temp ^ a[j], temp);
                    best = max(best, temp);
                    r = -1;
                    break;
                }
                if (bit_l == 0 && bit_r == 1) {
                    r = -1;
                    break;
                }
            }
        }
        if (r != -1) best = max(best, r);

        return best;
    }

    friend basis operator+(basis const &l, basis const &r) {
        if(l.sz == Log) return l;
        if(r.sz == Log) return r;
        auto res = l;
        for(int i = 0; i < Log; i++) if(r.a[i]) res.add(r.a[i]);
        return res;
    }
};
```
# N-SAT
```cpp {.numberLines}
struct nsat {
    int n;
    vector<vector<int>> lit, g;
    vector<int> occ, pos, neg, val, stk, lvl;

    nsat(int n) : n(n), g(2 * n), occ(2 * n), val(n, -1) {}

    // Pass a list of pairs: {variable_index, is_true}
    // Example: { {0, 1}, {1, 0} } -> (x_0 OR NOT x_1)
    void add_clause(initializer_list<pair<int, int>> vars) {
        vector<int> c;
        for (auto [u, is_true] : vars) {
            int L = (u << 1) | (!is_true);
            c.push_back(L);
            g[L].push_back(lit.size());
            occ[L]++;
        }
        lit.push_back(c);
    }

    void apply(int L) {
        val[L >> 1] = !(L & 1);
        stk.push_back(L);
        for (int c : g[L]) {
            if (pos[c]++ == 0)
                for (int u : lit[c]) occ[u]--;
        }
        for (int c : g[L ^ 1]) neg[c]++;
    }

    void undo() {
        int L = stk.back();
        stk.pop_back();
        val[L >> 1] = -1;
        for (int c : g[L]) {
            if (--pos[c] == 0)
                for (int u : lit[c]) occ[u]++;
        }
        for (int c : g[L ^ 1]) neg[c]--;
    }

    bool deduce(int &q_head) {
        while (q_head < stk.size()) {
            int L = stk[q_head++];
            for (int c : g[L ^ 1]) {
                if (pos[c] > 0) continue; // Clause already satisfied
                if (neg[c] == lit[c].size()) return false; // Conflict found

                if (neg[c] + 1 == lit[c].size()) { // Unit clause triggered
                    for (int u : lit[c]) {
                        if (val[u >> 1] == -1) {
                            apply(u);
                            break;
                        }
                    }
                }
            }
        }
        return true;
    }

    bool ok() {
        pos.assign(lit.size(), 0);
        neg.assign(lit.size(), 0);
        val.assign(n, -1);
        stk.clear();
        lvl.clear();
        int q_head = 0;

        // Force evaluate base unit clauses
        for (int c = 0; c < lit.size(); ++c) {
            if (lit[c].empty()) return false;
            if (lit[c].size() == 1) {
                int L = lit[c][0];
                if (val[L >> 1] == -1) apply(L);
                else if (val[L >> 1] != !(L & 1)) return false;
            }
        }

        // Sort variables by frequency for fast O(1) branching
        vector<int> order(n);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            return occ[a << 1] + occ[a << 1 | 1] > occ[b << 1] + occ[b << 1 | 1];
        });

        while (true) {
            if (deduce(q_head)) {
                // Find next unassigned variable to branch on
                int s = -1;
                for (int v : order) {
                    if (val[v] == -1) { s = v; break; }
                }

                // If all variables are assigned, we found a satisfying assignment
                if (s == -1) return true;

                lvl.push_back(stk.size());
                apply(s << 1); // Guess that variable 's' is True
            } else {
                // Conflict! Backtrack to last decision
                if (lvl.empty()) return false;

                int backtrack_idx = lvl.back();
                lvl.pop_back();

                int last_guess = stk[backtrack_idx];
                while (stk.size() > backtrack_idx) undo();

                q_head = stk.size();
                apply(last_guess ^ 1); // Flip the guess
            }
        }
    }
};
```
# Max Xor Subset In Range
```cpp {.numberLines}
// Given queries L,R find max XOR subset in range
const int N = 5e5 + 5;
vector<pair<int,int>>qqq[N];
int ans[N],basis[22],arr[N],last[22];

void add(int ind,int x){
    for (int i = 21; i >= 0; --i) {
        if((x >> i & 1) == 0)continue;
        if(ind > last[i]){
            swap(x,basis[i]);
            swap(ind,last[i]);
        }
        x ^= basis[i];
    }
}
int query(int ind){
    int ret = 0;
    for (int i = 21; i >= 0; --i) {
        if(last[i] >= ind){
            ret = max(ret,ret ^ basis[i]);
        }
    }
    return ret;
}

void solve() {
    int n;cin >> n;
    memset(last,-1,sizeof last);
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    }
    int q;cin >> q;
    for (int i = 0; i < q; ++i) {
        int l,r;cin >> l >> r;
        l--,r--;
        qqq[r].emplace_back(l,i);
    }
    for (int i = 0; i < n; ++i) {
        add(i,arr[i]);
        for(auto &x:qqq[i]){
            int l = x.fi;
            int ind = x.se;
            ans[ind] = query(l);
        }
    }
    for (int i = 0; i < q; ++i) {
        cout << ans[i] << endl;
    }
}
```
# and (&) in range [l, r]
```cpp {.numberLines}
ll andRange(ll l, ll r) {
    ll ans=0, msb = -1;
    while(r>0) { r >>= 1; msb++;}
    for(ll i= msb; ~i; i--) {
        ll la = 1;
        if((l&(la << i)) == (r&(la << i)))
            ans += (l&(la << i));
        else
            break;
    }
    return ans;
}
```
# Dynamic Bitset
```cpp {.numberLines}
template <int N>
struct Bitset {
    using T = uint64_t;
    static constexpr int sz_ = 64, shift = 6, AND = 63;
    static constexpr int M = (N + AND) >> shift;
    static constexpr T rem_ = (N % sz_ == 0) ? ~T(0) : (T(1) << (N % sz_)) - 1;

    array<T, M> b{};

    // ===== Basic operations =====
    inline void set() { fill(b.begin(), b.end(), ~T(0)); b.back() &= rem_; }
    inline void reset() { fill(b.begin(), b.end(), 0); }
    inline void flip() { for (T &i : b) i ^= ~T(0); b.back() &= rem_; }

    inline void set(int i) { b[i >> shift] |= T(1) << (i & AND); }
    inline void reset(int i) { b[i >> shift] &= ~(T(1) << (i & AND)); }
    inline void flip(int i) { b[i >> shift] ^= T(1) << (i & AND); }
    inline bool test(int i) const { return (b[i >> shift] >> (i & AND)) & 1; }
    inline bool operator[](int i) const { return test(i); }

    // ===== Queries =====
    inline bool any() const {
        for (T x : b) if (x) return true;
        return false;
    }
    inline bool none() const { return !any(); }
    inline bool all() const {
        for (int i = 0; i < M - 1; i++) if (~b[i]) return false;
        return b.back() == rem_;
    }
    inline int count() const {
        int res = 0;
        for (T x : b) res += __builtin_popcountll(x);
        return res;
    }

    // ===== In-place Bitwise ops (Optimized) =====
    Bitset& operator|=(const Bitset &a) { for (int i = 0; i < M; i++) 
        b[i] |= a.b[i]; return *this; }
    Bitset& operator&=(const Bitset &a) { for (int i = 0; i < M; i++) 
        b[i] &= a.b[i]; return *this; }
    Bitset& operator^=(const Bitset &a) { for (int i = 0; i < M; i++) 
        b[i] ^= a.b[i]; return *this; }
    
    Bitset operator|(const Bitset &a) const { return Bitset(*this) |= a; }
    Bitset operator&(const Bitset &a) const { return Bitset(*this) &= a; }
    Bitset operator^(const Bitset &a) const { return Bitset(*this) ^= a; }
    Bitset operator~() const { Bitset res = *this; res.flip(); return res; }

    bool operator==(const Bitset &a) const { return b == a.b; }
    bool operator!=(const Bitset &a) const { return b != a.b; }

    // ===== Shifts (Fixed & In-place optimized) =====
    Bitset& operator>>=(int n) {
        if (n >= N) { reset(); return *this; }
        int nxt = n >> shift, sh = n & AND, sh1 = sz_ - sh;
        for (int i = 0; i < M - nxt; i++) {
            b[i] = b[i + nxt] >> sh;
            if (sh && i + nxt + 1 < M) b[i] |= b[i + nxt + 1] << sh1;
        }
        fill(b.begin() + M - nxt, b.end(), 0);
        return *this;
    }
    
    Bitset& operator<<=(int n) {
        if (n >= N) { reset(); return *this; }
        int nxt = n >> shift, sh = n & AND, sh1 = sz_ - sh;
        for (int i = M - 1; i >= nxt; i--) {
            b[i] = b[i - nxt] << sh;
            if (sh && i - nxt - 1 >= 0) b[i] |= b[i - nxt - 1] >> sh1;
        }
        fill(b.begin(), b.begin() + nxt, 0);
        b.back() &= rem_;
        return *this;
    }

    Bitset operator>>(int n) const { return Bitset(*this) >>= n; }
    Bitset operator<<(int n) const { return Bitset(*this) <<= n; }

    // ===== Find methods =====
    inline int find_first() const {
        for (int i = 0; i < M; i++) if (b[i]) 
            return (i << shift) + __builtin_ctzll(b[i]);
        return -1;
    }
    
    inline int find_next(int pos) const {
        if (++pos >= N) return -1;
        int blk = pos >> shift;
        T mask = b[blk] & (~T(0) << (pos & AND));
        if (mask) return (blk << shift) + __builtin_ctzll(mask);
        for (int i = blk + 1; i < M; i++) if (b[i]) 
            return (i << shift) + __builtin_ctzll(b[i]);
        return -1;
    }
    
    inline int find_prev(int pos) const {
        if (--pos < 0) return -1;
        int blk = pos >> shift;
        T mask = b[blk] & (~T(0) >> (63 - (pos & AND)));
        if (mask) return (blk << shift) + 63 - __builtin_clzll(mask);
        for (int i = blk - 1; i >= 0; i--) {
            if (b[i]) return (i << shift) + 63 - __builtin_clzll(b[i]);
        }
        return -1;
    }
};
```
# Bit Twiddle Permute
```cpp {.numberLines}
int bit_twiddle_permute(int v) { // next integer that has _pop_count(v) bits
    int t = v | (v - 1);
    int w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
    return w;
}
```