template<class T>
class mex_calculator {
    map<T, T> count;
    set<T> missing;
    public:
    mex_calculator(const vector<T>& arr, T upper_bound) { // n log n
        for (T x : arr)
            count[x]++;
        for (T i = 0; i <= upper_bound + 1; ++i)
            if (count[i] == 0)
                missing.insert(i);
    }
    void insert(T x) { // log
        count[x]++;
        if (count[x] == 1)
            missing.erase(x);
    }
    void remove(T x) { // log
        if (count[x] == 0) return;
        count[x]--;
        if (count[x] == 0)
            missing.insert(x);
    }
    T get_mex() { // 1
        return *missing.begin();
    }
};