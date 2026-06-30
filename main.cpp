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


template <typename T>
struct Matrix {
    int n, m;
    vector<T> mat;

    Matrix(int n, int m) : n(n), m(m) {
        mat.assign(n * m, 0);
    }
    Matrix(int n) : n(n), m(n) {
        mat.assign(n * n, 0);
    }

    void make_unit() {
        assert(n == m);
        fill(mat.begin(), mat.end(), 0);
        for (int i = 0; i < n; i++) {
            mat[i * m + i] = 1;
        }
    }

    T* operator[](int r) { return &mat[r * m]; }
    const T* operator[](int r) const { return &mat[r * m]; }

    Matrix operator*(const Matrix& o) const {
        assert(m == o.n);
        Matrix res(n, o.m);

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < m; k++) {
                T val = mat[i * m + k];
                if (val == 0) continue;

                for (int j = 0; j < o.m; j++) {
                    res[i][j] = res[i][j] + val * o[k][j];
                }
            }
        }
        return res;
    }

    Matrix operator+(const Matrix& o) const {
        assert(o.n == n && o.m == m);
        Matrix ret(n, m);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                ret[i][j] = (*this)[i][j] + o[i][j];
        return ret;
    }

    Matrix pow(long long k) const {
        assert(n == m);
        Matrix res(n), base = *this;
        res.make_unit();
        while (k > 0) {
            if (k & 1) res = res * base;
            base = base * base;
            k >>= 1;
        }
        return res;
    }
};

void solve() {

}