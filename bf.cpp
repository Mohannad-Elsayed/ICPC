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
    int x = mt19937(random_device()())();
    print(x);
    freopen("out", "wt", stdout);
#else
    cin.tie(0)->sync_with_stdio(0);
#endif
    cin.exceptions(cin.failbit);
    int t = 1;
    // cin >> t;
    while (t--) {
        solve();
        if (t) cout << '\n';
    }
    return 0;
}

const int MOD = 1e9 + 7;

template<typename T>
T inverse(T a, T m) {
    T u = 0, v = 1;
    while (a != 0) {
        T t = m / a;
        m -= t * a;
        swap(a, m);
        u -= t * v;
        swap(u, v);
    }
    // assert(m == 1);
    if (m != 1) return 1;
    return u;
}

template<typename T>
class Modular {
public:
    using Type = typename decay<decltype(T::value)>::type;

    constexpr Modular() : value() {
    }

    template<typename U>
    Modular(const U &x) {
        value = normalize(x);
    }

    template<typename U>
    static Type normalize(const U &x) {
        Type v;
        if (-mod() <= x && x < mod()) v = static_cast<Type>(x);
        else v = static_cast<Type>(x % mod());
        if (v < 0) v += mod();
        return v;
    }

    const Type &operator()() const { return value; }

    template<typename U>
    explicit operator U() const { return static_cast<U>(value); }

    constexpr static Type mod() { return T::value; }

    Modular &operator+=(const Modular &other) {
        if ((value += other.value) >= mod()) value -= mod();
        return *this;
    }

    Modular &operator-=(const Modular &other) {
        if ((value -= other.value) < 0) value += mod();
        return *this;
    }

    template<typename U>
    Modular &operator+=(const U &other) { return *this += Modular(other); }

    template<typename U>
    Modular &operator-=(const U &other) { return *this -= Modular(other); }

    Modular &operator++() { return *this += 1; }
    Modular &operator--() { return *this -= 1; }

    Modular operator++(int) {
        Modular result(*this);
        *this += 1;
        return result;
    }

    Modular operator--(int) {
        Modular result(*this);
        *this -= 1;
        return result;
    }

    Modular operator-() const { return Modular(-value); }

    template<typename U = T>
    typename enable_if<is_same<typename Modular<U>::Type, int>::value, Modular>::type &operator*=(const Modular &rhs) {
        value = normalize(static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value));
        return *this;
    }

    template<typename U = T>
    typename enable_if<is_same<typename Modular<U>::Type, int64_t>::value, Modular>::type &operator*=(
        const Modular &rhs) {
        int64_t q = int64_t(static_cast<long double>(value) * rhs.value / mod());
        value = normalize(value * rhs.value - q * mod());
        return *this;
    }

    template<typename U = T>
    typename enable_if<!is_integral<typename Modular<U>::Type>::value, Modular>::type &operator*=(const Modular &rhs) {
        value = normalize(value * rhs.value);
        return *this;
    }

    Modular &operator/=(const Modular &other) { return *this *= Modular(inverse(other.value, mod())); }

    friend const Type &abs(const Modular &x) { return x.value; }

    template<typename U>
    friend bool operator==(const Modular<U> &lhs, const Modular<U> &rhs);

    template<typename U>
    friend bool operator<(const Modular<U> &lhs, const Modular<U> &rhs);

    template<typename V, typename U>
    friend V &operator>>(V &stream, Modular<U> &number);

private:
    Type value;
};

template<typename T>
bool operator==(const Modular<T> &lhs, const Modular<T> &rhs) { return lhs.value == rhs.value; }

template<typename T, typename U>
bool operator==(const Modular<T> &lhs, U rhs) { return lhs == Modular<T>(rhs); }

template<typename T, typename U>
bool operator==(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) == rhs; }

template<typename T>
bool operator!=(const Modular<T> &lhs, const Modular<T> &rhs) { return !(lhs == rhs); }

template<typename T, typename U>
bool operator!=(const Modular<T> &lhs, U rhs) { return !(lhs == rhs); }

template<typename T, typename U>
bool operator!=(U lhs, const Modular<T> &rhs) { return !(lhs == rhs); }

template<typename T>
bool operator<(const Modular<T> &lhs, const Modular<T> &rhs) { return lhs.value < rhs.value; }

template<typename T>
Modular<T> operator+(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) += rhs; }

template<typename T, typename U>
Modular<T> operator+(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) += rhs; }

template<typename T, typename U>
Modular<T> operator+(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) += rhs; }

template<typename T>
Modular<T> operator-(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) -= rhs; }

template<typename T, typename U>
Modular<T> operator-(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) -= rhs; }

template<typename T, typename U>
Modular<T> operator-(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) -= rhs; }

template<typename T>
Modular<T> operator*(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) *= rhs; }

template<typename T, typename U>
Modular<T> operator*(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) *= rhs; }

template<typename T, typename U>
Modular<T> operator*(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) *= rhs; }

template<typename T>
Modular<T> operator/(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) /= rhs; }

template<typename T, typename U>
Modular<T> operator/(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) /= rhs; }

template<typename T, typename U>
Modular<T> operator/(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) /= rhs; }

template<typename T, typename U>
Modular<T> power(const Modular<T> &a, const U &b) {
    assert(b >= 0);
    Modular<T> x = a, res = 1;
    U p = b;
    while (p > 0) {
        if (p & 1) res *= x;
        x *= x;
        p >>= 1;
    }
    return res;
}

template<typename T>
bool IsZero(const Modular<T> &number) {
    return number() == 0;
}

template<typename T>
string to_string(const Modular<T> &number) {
    return to_string(number());
}

// U == std::ostream? but done this way because of fastoutput
template<typename U, typename T>
U &operator<<(U &stream, const Modular<T> &number) {
    return stream << number();
}

// U == std::istream? but done this way because of fastinput
template<typename U, typename T>
U &operator>>(U &stream, Modular<T> &number) {
    typename common_type<typename Modular<T>::Type, int64_t>::type x;
    stream >> x;
    number.value = Modular<T>::normalize(x);
    return stream;
}

constexpr int md = MOD;
using Mint = Modular<std::integral_constant<decay<decltype(md)>::type, md> >;

int n;
vector<vector<int> > tree;
vector<vector<Mint> > dp, pfx, sfx;
vector<Mint> ans;

void dfs1(int u, int p) {
    vector<int> nodes;
    Mint p1, p2;
    p1 = p2 = 1;
    for (auto v: tree[u])
        if (v != p) {
            nodes.push_back(v);
            dfs1(v, u);
            dp[u][0] *= dp[v][0];
            dp[u][1] *= dp[v][0] + dp[v][1];
            pfx[v][0] = p1;
            pfx[v][1] = p2;
            p1 *= dp[v][0];
            p2 *= dp[v][0] + dp[v][1];
        }

    p1 = p2 = 1;
    for (int i = nodes.size() - 1; ~i; i--) {
        sfx[nodes[i]][0] = p1;
        sfx[nodes[i]][1] = p2;
        p1 *= dp[nodes[i]][0];
        p2 *= dp[nodes[i]][1] + dp[nodes[i]][0];
    }
}

void dfs2(int u, int p) {
    ans[u] = dp[u][1];
    auto ou = dp[u];
    auto op = pfx[u];
    auto os = sfx[u];
    Mint p00 = ~p ? dp[p][0] : 1;
    Mint p01 = ~p ? dp[p][1] : 1;
    for (auto c : tree[u])
        if (c != p) {

            auto oc = dp[c];
            // move root to c
            // remove c from u subtree
            dp[u][0] = pfx[c][0] * sfx[c][0];
            dp[u][1] = pfx[c][1] * sfx[c][1];
            // calculate the root for c
            dp[c][0] *= dp[u][0] * p00;
            dp[c][1] *= dp[u][1] + dp[u][0];
            dp[c][1] *= p01;
            // u now is
            // recurse
            dfs2(c, u);
            // move root back to u
            dp[c] = oc;
            dp[u] = ou;
            pfx[u] = op;
            sfx[u] = os;
        }
}

void solve() {
    cin >> n;
    tree.resize(n);
    ans.resize(n);
    dp = vector(n, vector(2, Mint(1)));
    pfx = sfx = dp;
    for (int i = 1; i < n; i++) {
        int x;
        cin >> x;
        x--;
        tree[x].push_back(i);
        tree[i].push_back(x);
    }

    dfs1(0, -1);
    print(dp);
    dfs2(0, -1);
    print(dp);
    for (auto x: ans) cout << x << ' ';
}
