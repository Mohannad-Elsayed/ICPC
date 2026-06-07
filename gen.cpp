#include "testlib.h"
#include <set>
#include <vector>
#include <algorithm>

using namespace std;

const int MAX_COORD = 1'000'000'000;

void generateTestCase(int n, int type) {
    set<pair<int, int>> used;
    vector<pair<int, int>> asteroids;

    auto get_unique_pt = [&](int min_x, int max_x, int min_y, int max_y) {
        while (true) {
            int x = rnd.next(min_x, max_x);
            int y = rnd.next(min_y, max_y);
            if (used.insert({x, y}).second) return make_pair(x, y);
        }
    };

    // Generate the Ship first
    pair<int, int> ship;

    if (type == 0) {
        // Pure Random
        ship = get_unique_pt(1, MAX_COORD, 1, MAX_COORD);
        for (int i = 0; i < n; ++i) {
            asteroids.push_back(get_unique_pt(1, MAX_COORD, 1, MAX_COORD));
        }
    }
    else if (type == 1) {
        // All asteroids on different rows
        ship = get_unique_pt(1, MAX_COORD / 2, 1, MAX_COORD);
        for (int i = 0; i < n; ++i) {
            int x = rnd.next(MAX_COORD / 2 + 1, MAX_COORD);
            asteroids.push_back(get_unique_pt(x, x, 1, MAX_COORD));
        }
    }
    else if (type == 2) {
        // All asteroids on the same row, but BEHIND the ship
        ship = get_unique_pt(1, MAX_COORD, 1, 100'000);
        for (int i = 0; i < n; ++i) {
            asteroids.push_back(get_unique_pt(ship.first, ship.first, 100'001, MAX_COORD));
        }
    }
    else if (type == 3) {
        // Maximum shielding: ALL asteroids are on the same row, IN FRONT of the ship
        ship = get_unique_pt(1, MAX_COORD, MAX_COORD - 100'000, MAX_COORD);
        for (int i = 0; i < n; ++i) {
            asteroids.push_back(get_unique_pt(ship.first, ship.first, 1, ship.second - 1));
        }
    }

    println(n);
    for (auto p : asteroids) {
        println(p.first, p.second);
    }
    println(ship.first, ship.second);
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);

    int t = opt<int>("t", rnd.next(1, 2000));
    int sumn = opt<int>("sumn", 2000);
    int type = opt<int>("type", 0);
    // Types: 0: Random, 1: No ast on ship row, 2: Ast behind ship, 3: Max shielding

    println(t);
    vector<int> n_len = rnd.partition(t, sumn, 0); // Allows n=0

    for (int tt = 0; tt < t; tt++) {
        setTestCase(tt + 1);
        generateTestCase(n_len[tt], type);
    }

    return 0;
}