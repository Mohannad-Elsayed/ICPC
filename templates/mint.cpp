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

    ModInt &operator+=(ModInt o) { v = (v + o.v) % M; return *this; }
    ModInt &operator-=(ModInt o) { v = (v - o.v + M) % M; return *this; }
    ModInt &operator*=(ModInt o) { v = (1ll * v * o.v) % M; return *this; }
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
using mint = ModInt<>;
