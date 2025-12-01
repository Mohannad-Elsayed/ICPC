#include "testlib.h"
#include <vector>
#include <algorithm>

using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);

    // Parse command line arguments with defaults
    int q = opt<int>("q", 5000);
    int k = opt<int>("k", 5000);
    int max_x = opt<int>("mxx", 5000);
    int add_prob = opt<int>("ap", 60); // Probability of adding vs removing

    // Output first line
    println(q, k);

    // Track existing balls to ensure valid Type 2 operations
    vector<int> box;

    for (int i = 0; i < q; ++i) {
        int t;

        // If box is empty, we MUST add (Type 1)
        if (box.empty()) {
            t = 1;
        } else {
            // Otherwise, decide based on probability
            // If random value [0, 99] < add_prob, we add.
            if (rnd.next(0, 99) < add_prob) {
                t = 1;
            } else {
                t = 2;
            }
        }

        if (t == 1) {
            // Operation 1: Add x
            // We usually want x <= k to be useful, but x <= 5000 is the constraint.
            // Let's mix it up: 90% chance x <= k, 10% chance x <= max_x
            int x;
            if (k < max_x && rnd.next(0, 9) > 0) {
                 x = rnd.next(1, k);
            } else {
                 x = rnd.next(1, max_x);
            }

            println(1, x);
            box.push_back(x);
        } else {
            // Operation 2: Remove x
            // Must pick an x that exists in the box
            int index = rnd.next(0, (int)box.size() - 1);
            int x = box[index];

            println(2, x);

            // Remove from our internal tracker efficiently
            // Swap with back and pop
            box[index] = box.back();
            box.pop_back();
        }
    }

    return 0;
}