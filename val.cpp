#include "testlib.h"
#include <set>

using namespace std;

const int MAX_SUM_N = 2000;
const int MAX_COORD = 1'000'000'000;

int main(int argc, char* argv[]) {
    registerValidation(argc, argv);

    int t = inf.readInt(1, 2000, "t");
    inf.readEoln();

    int sum_n = 0;

    for (int test = 1; test <= t; test++) {
        setTestCase(test);

        int n = inf.readInt(0, 2000, "n");
        inf.readEoln();

        sum_n += n;
        ensuref(sum_n <= MAX_SUM_N, "Sum of n exceeds %d", MAX_SUM_N);

        set<pair<int, int>> used_coords;

        // Read Asteroids
        for (int i = 0; i < n; i++) {
            int a = inf.readInt(1, MAX_COORD, "a_i");
            inf.readSpace();
            int b = inf.readInt(1, MAX_COORD, "b_i");
            inf.readEoln();

            ensuref(used_coords.insert({a, b}).second,
                    "Asteroid coordinates must be distinct. Duplicate found at (%d, %d)", a, b);
        }

        // Read Ship
        int x_s = inf.readInt(1, MAX_COORD, "x_s");
        inf.readSpace();
        int y_s = inf.readInt(1, MAX_COORD, "y_s");
        inf.readEoln();

        ensuref(used_coords.insert({x_s, y_s}).second,
                "Ship coordinates must be distinct from asteroids. Conflict at (%d, %d)", x_s, y_s);
    }

    inf.readEof();
    return 0;
}