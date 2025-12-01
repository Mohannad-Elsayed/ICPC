#include "bits/stdc++.h"
using namespace std;
#define N 100100
#define B 430
#define NB (N/B+2)

int a[N], n, q;
long long Ans[N];
long long cans = 0;

typedef struct Q {
    int l, r, idx;
    bool operator<(const Q& o) const {
        return l/B == o.l/B ? l/B&1 ? r > o.r : r < o.r : l < o.l;
    }
} query;

map<int, int> mp;
set<pair<int, int>> ste;
int f;

void add(int i) {
    auto it = mp.find(i);
    if (it == mp.end()) {
        mp[i]++;
        ste.emplace(1, i);
        return;
    }
    f = it->second;
    ste.erase({f, i});
    ste.emplace(f+1, i);
    mp[i]++;
}

void rem(int i) {
    auto it = mp.find(i);
    f = it->second;
    ste.erase({f, i});
    f--;
    if (!f) {
        mp.erase(i);
        return;
    }
    mp[i]--;
    ste.emplace(f, i);
}

int main() {
    scanf("%d%d", &n, &q);
    for (int i = 0; i < n; i++) {
        scanf("%d", a+i);
    }
    query qr[q];
    for (int i = 0; i < q; i++) {
        scanf("%d%d", &qr[i].l, &qr[i].r);
        qr[i].idx = i;
    }
    sort(qr, qr+q);
    int l = 0, r = -1;
    for (int i = 0, ll, rr; i < q; i++) {
        ll = qr[i].l;
        rr = qr[i].r;
        while (l > ll) add(a[--l]);
        while (r < rr) add(a[++r]);
        while (l < ll) rem(a[l++]);
        while (r > rr) rem(a[r--]);
        // print(ll, rr, ste);
        Ans[qr[i].idx] = ste.rbegin()->first;
    }
    for (int i = 0; i < q; i++)
        printf("%lld\n", Ans[i]);
}