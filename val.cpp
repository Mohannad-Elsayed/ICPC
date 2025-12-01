#include "testlib.h"
#include <vector>

using namespace std;

const int MAX_VAL = 5000;

int main(int argc, char* argv[]) {
    registerValidation(argc, argv);

    // Read q and k
    int q = inf.readInt(1, 5000, "q");
    inf.readSpace();
    inf.readInt(1, 5000, "k");
    inf.readEoln();

    // Frequency array to track balls currently in the box
    // Used to validate the removal guarantee
    vector<int> counts(MAX_VAL + 1, 0);

    for (int i = 0; i < q; ++i) {
        setTestCase(i+1);
        int t = inf.readInt(1, 2, "t");
        inf.readSpace();

        int x = inf.readInt(1, MAX_VAL, "x");
        inf.readEoln();

        if (t == 1) {
            counts[x]++;
        } else {
            // Remove ball
            // Check guarantee: box must contain x
            ensuref(counts[x] > 0, "Query %d: Operation '2 %d' requested, but ball %d is not in the box.", i + 1, x, x);
            counts[x]--;
        }
    }

    inf.readEof();
    return 0;
}