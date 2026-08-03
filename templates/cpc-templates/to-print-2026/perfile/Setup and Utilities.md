---
title: 1. Setup and Utilities
---
# Template
```cpp {.numberLines}
#include "bits/stdc++.h"
using namespace std;
using ll = long long;
#define int ll
#define endl '\n'
void solve() {
}
signed main() {
    cin.tie(0)->sync_with_stdio(0);
    cin.exceptions(cin.failbit);
    int tt = 1;
    // cin >> tt;
    while (tt--) {
        solve();
        cout << '\n';
    }
    return 0;
}
```
# Precompile stdc++.h
```cpp {.numberLines}
sudo g++ -x c++-header -std=c++17 -O0 stdc++.h -o stdc++.h.gch
ulimit -s ${size in kb}
add empty template to live templates
increase undo range to 10k (ctrl+shift+A) -> registry -> undo
```
# Pragmas
```cpp {.numberLines}
#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
```
# Random mt19937
```cpp {.numberLines}
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
ll rnd(ll l, ll r) {
    static mt19937_64 gen(chrono::steady_clock::now().time_since_epoch().count());
    return uniform_int_distribution<ll>(l, r)(gen);
}
```
# Direction Arrays
```cpp {.numberLines}
int dx[8] = { 2, 1, -1, -2, -2, -1, 1, 2 };
int dy[8] = { 1, 2, 2, 1, -1, -2, -2, -1 }; // knight

int dx[8] = {-1,0,1,-1,1,-1,0,1};
int dy[8] = {-1,-1,-1,0,0,1,1,1}; // king

int dx[4] = {1, -1, 0, 0};
int dy[4] = {0, 0, -1, 1};
string direction = "DULR";
```
# Custom Comparator
```cpp {.numberLines}
struct cmp {
    bool operator() (int a, int b) const {
        return ...;
    }
};
set<int, cmp> s;
```
# Fast HashMap (gp_hash_table)
```cpp {.numberLines}
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
struct chash {
    const int RANDOM = (ll)(make_unique<char>().get()) ^
        chrono::high_resolution_clock::now().time_since_epoch().count();
    
    static unsigned long long hash_f(unsigned long long x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    static unsigned hash_combine(unsigned a, unsigned b) {
        return a * 31 + b;
    }
    int operator()(int x) const {
        return hash_f(x)^RANDOM;
    }
};
gp_hash_table<int, int, chash> table;
```
# __int128_t
```cpp {.numberLines}
istream& operator>>(istream& is, __int128_t& v) {
    string s; is >> s; v = 0;
    for (char c : s) if (isdigit(c)) v = v * 10 + c - '0';
    if (s[0] == '-') v = -v;
    return is;
}
ostream& operator<<(ostream& os, __int128_t v) {
    if (!v) return os << "0";
    if (v < 0) os << '-', v = -v;
    string s;
    while (v) s += '0' + v % 10, v /= 10;
    reverse(s.begin(), s.end());
    return os << s;
}
```