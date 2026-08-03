#include "bits/stdc++.h"
using namespace std;

#define endl '\n'
#define ll long long
#define ull unsigned long long
#define ld long double

#define BegEnd(num) num.begin(), num.end()
#define RBegEnd(num) num.rbegin(), num.rend()
#define getVec(arr, size)   \
    vector<int> arr(size);  \
    for (auto &input : arr) cin >> input;

#define mem(arr, val) memset(arr, val, sizeof(arr))

#define print(z, n)                                            \
    for (int i = 0; (n && i < n) || (!n && i < z.size()); i++) \
        cout << z[i] << ' ';                                   \
    cout << endl;

#define pii pair<int, int>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vpii vector<pair<int, int>>

#define F first
#define S second
//#define int ll
#define FIO { ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr); }

const int M = 1e9 + 7, OO = 1e9;

int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
int dy[] = {0, 0, 1, -1, -1, 1, -1, 1};
string dd[] = {"U", "D", "R", "L", "UL", "UR", "DL", "DR"};

struct TrieString {
    struct Node {
        int child[10], prefixCount = 0;
        bool end = false;
        Node() {
            fill(child, child + 10, -1);
        }
    };

    vector<Node> tr;
    TrieString() {
        tr.push_back(Node());
    }

    void insert(const string& s) {
        int cur = 0;
        for (char ch : s) {
            int idx = ch;
            if (tr[cur].child[idx] == -1) {
                tr[cur].child[idx] = (int)tr.size();
                tr.emplace_back();
            }
            cur = tr[cur].child[idx];
            tr[cur].prefixCount++;
        }
    }

    string getmx(const string& s) {
        int cur = 0;
        string ans;
        for (char ch : s) {
            int z = 10 - ch;
            for (int i = z-1, j = 0; j < 10; j++) {
                i += 10;
                i %= 10;

                if (tr[cur].child[i] != -1) {
                    ans.push_back((i+ch)%10+'0');
                    cur = tr[cur].child[i];
                    break;
                }

                i--;
            }
        }
        return ans;
    }

    string getmn(const string& s) {
        int cur = 0;
        string ans;
        for (char ch : s) {
            int z = 10 - ch;
            for (int i = z, j = 0; j < 10; j++) {
                i += 10;
                i %= 10;

                if (tr[cur].child[i] != -1) {
                    ans.push_back((i+ch)%10+'0');
                    cur = tr[cur].child[i];
                    break;
                }

                i++;
            }
        }
        return ans;
    }


    void erase(const string& s) {
        int cur = 0;

        for (char ch : s) {
            int idx = ch;
            if (tr[cur].child[idx] == -1) return;

            int next = tr[cur].child[idx];
            tr[next].prefixCount--;

            if(tr[next].prefixCount == 0){
                tr[cur].child[idx] = -1;
                return;
            }

            cur = next;
        }
    }
};

void solve() {
    int n;
    cin >> n;

    vector<string> num(n);
    for(int i = 0; i < n; i++){
        string s;
        cin >> s;
        while(s.size() < 19){
            s = '0' + s;
        }
        num[i] = s;
    }
    for (auto &s : num)
        for (auto &ch : s)
            ch -= '0';
    TrieString trie;
    for (auto &x : num) trie.insert(x);

    string mx, mn(20, '9');
    for(int i = 0; i < n; i++){
        if (i)
        mx = max(mx, trie.getmx(num[i])),
        mn = min(mn, trie.getmn(num[i]));
        trie.insert(num[i]);
    }

    int go = 0;
    for(int i = 0; i < mn.size(); i++){
        if(mn[i] != '0'){
            go = true;
        }

        if(go) cout << mn[i];
    }
    if(!go) cout << 0;
    cout << ' ';

    go = 0;
    for(int i = 0; i < mx.size(); i++){
        if(mx[i] != '0'){
            go = true;
        }

        if(go) cout << mx[i];
    }
    if(!go) cout << 0;
}

signed main()
{
    FIO

    int t = 1;
//    cin >> t;
    for (int i = 1; i <= t; i++)
    {
        solve();
        cout << endl;
    }
    // cerr << clock() / 1000.0 << " Secs";
}

// ####################
// ##### 3BcarenO #####
// ####################