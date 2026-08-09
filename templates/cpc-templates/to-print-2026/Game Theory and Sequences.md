---
title: 8. Game Theory and Sequences
---
# Mex Calculator
```cpp {.numberLines}
class mex_calculator {
    map<int, int> count;
    set<int> missing;
    public:
    mex_calculator(const vector<int>& arr, int upper_bound) { // n log n
        for (int x : arr)
            count[x]++;
        for (int i = 0; i <= upper_bound + 1; ++i)
            if (count[i] == 0)
                missing.insert(i);
    }
    void insert(int x) { // log
        count[x]++;
        if (count[x] == 1)
            missing.erase(x);
    }
    void remove(int x) { // log
        if (count[x] == 0) return;
        count[x]--;
        if (count[x] == 0)
            missing.insert(x);
    }
    int get_mex() { // 1
        return *missing.begin();
    }
};
```
# Remove Game
```cpp {.numberLines}
void solve() {
    int n; cin >> n;
    vector<int> v(n);
    getv(v);
    ll diff[n+1][n+1]{}; // mx diff p1-p2 on interval [l, r]
    for (int l = n-1; ~l; l--) {
        for (int r = 0; r < n; r++) {
            // dp[l][r] = mx( v[l]-dp[l+1][r], v[r]-dp[l][r-1] )
            if (l == r) 
                diff[l][r] = v[l];
            else diff[l][r] = max<ll>(v[l] - diff[l+1][r], v[r]-diff[l][r-1]);
        }
    }

    cout << (accumulate(all(v), 0ll) + diff[0][n-1])/2;
}
```
# K-th Balanced Bracket Sequence
```cpp {.numberLines}
//O(n^2)
string kth_balanced(int n, int k) {
    vector<vector<int>> d(2*n+1, vector<int>(n+1, 0));
    d[0][0] = 1;
    for (int i = 1; i <= 2*n; i++) {
        d[i][0] = d[i-1][1];
        for (int j = 1; j < n; j++)
            d[i][j] = d[i-1][j-1] + d[i-1][j+1];
        d[i][n] = d[i-1][n-1];
    }
    string ans;
    int depth = 0;
    for (int i = 0; i < 2*n; i++) {
        if (depth + 1 <= n && d[2*n-i-1][depth+1] >= k)
        {
            ans += '(';
            depth++;
        } else {
            ans += ')';
            if (depth + 1 <= n)
                k -= d[2*n-i-1][depth+1];
            depth--;
        }
    }
    return ans;
}
```
# Next Balanced Sequence
```cpp {.numberLines}
bool next_balanced_sequence(string & s) { // O(n)
    int n = s.size();
    int depth = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (s[i] == '(')
            depth--;
        else
            depth++;
        if (s[i] == '(' && depth > 0) {
            depth--;
            int open = (n - i - 1 - depth) / 2;
            int close = n - i - 1 - open;
            string next = s.substr(0, i) + ')' +
            string(open, '(') + string(close, ')');
            s.swap(next);
            return true;
        }
    }
    return false;
}
```