---
title: 9. Geometry
---
# Geometry Notes
```cpp {.numberLines}
Generate 2 points on a line
// ax + by + c = 0
if(a == 0){
        p1 ={0,-1.0 * c/b};
        p2 = {1,-1.0 * c/b};
    }
    else{
        p1 = {-1.0* c/a,0};
        p2 = {-1.0 * (c + b)/a,1};
    }


You’re given 4 integers which are the
coefficients A B and C of the normal equation of the
straight line and a distance value R.

void solve() {
    double a,b,c,r;cin >> a >> b >> c >> r;
    double base = sqrt(a * a + b * b);
    a /= base;
    b /= base;
    c /= base;
    cout << setprecision(15) << a << ' ' << b << ' ' << c + r << endl;
    cout << setprecision(15) << a << ' ' << b << ' ' << c - r << endl;
}

Area of the sector without an angle = (l * r) / 2
The length of the arc l = (theta / 360) * 2 * pi * r

The area of the parallelogram = the cross-product of 
    2 adjacent sides = 2 * area of the triangle made by 3 points.

Given 1 side of the Pythagorean Triangle ... Get the missing 2 sides :
ll n;
cin >> n;
if (n == 1 || n==2)
    cout << -1;
else if (n & 1)
    cout << (n * n + 1) / 2 << " " << (n * n - 1) / 2;
else
    cout << n * n / 4 + 1 << " " << n * n / 4 - 1;

In triangle abc angle bac cos(theta) = (b^2 + c^2 - a^2) / (2 * b * c)

n = number of sides of a regular polygon
S = side length of the polygon
ap = apothem the distance from the center of the polygon to the middle of any side
r = radius of the polygon which is the distance from the center of the polygon to any corner.
p = perimeter of the polygon

p = S * n
ap = S / (2 * tan(180/n)) = r * cos(180/n)
r = S / (2 * sin(180/n)) = ap / cos(180/n)
Area = (p * ap)/2 , (S^2 * n) / (4 * tan(180/n)) = ap^2 * n * tan(180/n) 
    = (r^2 * n * sin(360/n))/2

sin(2*theta) = 2 * sin(theta) * cos(theta)
cos(2*theta) = cos(theta)^2 - sin(theta)^2 = 2 * cos(theta)^2 - 1 
    = 1 - 2 * sin(2*theta)^2
sin(theta)^2 = (1 - cos(2 * theta))/2
cos(theta)^2 = (1 + cos(2 * theta))/2
tan(2*theta) = (2 * tan(theta)) / (1 - tan(theta)^2)

Circle intersection r1,r2,d where r1 >= r2
If d = r1 + r2 they touch from outside
If d = r1 - r2 they touch from inside
If r1 - r2 < d < r1 + r2 they intersect in two points

Plane equation ax + by + cz + d = 0
AB = (Bx-Ax,By-Ay,Bz-Az)
AC = (Cx-Ax,Cy-Ay,Cz-Az)
AB x AC = (a,b,c)
a = (By-Ay)*(Cz-Az)-(Cy-Ay)*(Bz-Az)
b = (Bz-Az)*(Cx-Ax)-(Cz-Az)*(Bx-Ax)
c = (Bx-Ax)*(Cy-Ay)-(Cx-Ax)*(By-Ay)
d = -(a*Ax+b*Ay+c*Az)

// Checks if four points lie in the same plane or not
bool samePlane(point a,point b,point c){
    // a * (b x c) = volume = 0
    return (a.dot(b.cross(c)) == 0);
}

void solve() {
    vector<point>v(4);
    for (int i = 0; i < 4; ++i) {
        cin >> v[i].x >> v[i].y >> v[i].z;
    }
    for (int i = 0; i < 4; ++i) {
        v[i].x -= v[3].x;
        v[i].y -= v[3].y;
        v[i].z -= v[3].z;
    }
    cout << (samePlane(v[0],v[1],v[2]) ? "YES":"NO") << endl;
}
```
# Geometry (X, Y, dot, cross)
```cpp {.numberLines}
/*
 conj(a) -> a.imag() *= -1
 abs(point) distance between (0,0) to this point
 norm(point) squared magnitude -> real² + imag²
 hypot(x, y) -> sqrt(x² + y²)
 arg(vector) angle between this vector and x-axis
 clamp(a, l, r) == min(r, max(l, a))
 polar(rho, theta) -> make vector with length rho and angle theta
 internal angle = (n - 2) * 180 / n
 number of diagonals n * (n - 3) / 2
 Area(p) = internal_points_cnt + (boundary_points/2) - 1
 boundary_point in vector = gcd(|x2-x1|, |y2-y1|) + 1
 line have infinity point, segment have to end points
 vector(x, y) perpendicular to vector(-y, x) and (y, -x)
*/

using ll = int64_t;

using ld = double;
using pt = complex<ld>;

const ll INF = 7e18;
const ld EPS = 1e-9;
const ld PI = acos(-1);

#define X real()
#define Y imag()

#define dot(a, b) (conj(a) * (b)).X
#define cross(a, b) (conj(a) * (b)).Y

int sign(ld x) {
    return (x > EPS) - (x < -EPS);
}

struct compX{
    bool operator()(pt a, pt b) const {
        return a.X != b.X ? a.X < b.X : a.Y < b.Y;
    }
};
struct compY{
    bool operator()(pt a, pt b) const {
        return a.Y != b.Y ? a.Y < b.Y : a.X < b.X;
    }
};

// ================ line, segment ==========================

// projection of pt p onto line ab
pt project(pt a, pt b, pt p) {
    pt ab = b - a;
    return a + ab * dot(p - a, ab) / norm(ab);
}

// works for any orientation
bool onSegment(pt a, pt b, pt p) {
    return sign(cross(b - a, p - a)) == 0 &&
           sign(dot(p - a, p - b)) <= 0;
}

// ccw: >0 left, <0 right, =0 collinear
int ccw(pt a, pt b, pt c) {
    return sign(cross(b - a, c - a));
}

// works for any pts
ld distanceToLine(pt a, pt b, pt p) {
    return fabsl(cross(b - a, p - a)) / abs(b - a);
}

// works for any line
ld distanceToLine(ld A, ld B, ld C, pt p) {
    return fabsl(A*p.X + B*p.Y + C) / abs(pt(A, B));
}

// works for any pts
ld distanceToSegment(pt a, pt b, pt p) {
    if (dot(b - a, p - a) < 0) return abs(p - a);
    if (dot(a - b, p - b) < 0) return abs(p - b);
    return distanceToLine(a, b, p);
}

// works for intersecting lines (not parallel)
pt lineIntersect(pt a, pt b, pt c, pt d) {
    pt ab = b - a, cd = d - c;
    return a + ab * (cross(c - a, cd) / cross(ab, cd));
}

// works for all segments (returns intersection pt if exists)
bool segmentsIntersect(pt a, pt b, pt c, pt d, pt &inter) {
    int d1 = ccw(a, b, c), d2 = ccw(a, b, d);
    int d3 = ccw(c, d, a), d4 = ccw(c, d, b);

    if(d1 * d2 < 0 && d3 * d4 < 0)
        return inter = lineIntersect(a, b, c, d), true;

    if(d1 == 0 && onSegment(a, b, c)) return inter = c, true;
    if(d2 == 0 && onSegment(a, b, d)) return inter = d, true;
    if(d3 == 0 && onSegment(c, d, a)) return inter = a, true;
    if(d4 == 0 && onSegment(c, d, b)) return inter = b, true;

    return false;
}

// works for any triangle
ld triangleArea(pt a, pt b, pt c) {
    return 0.5 * fabs(cross(b - a, c - a));
}

bool ptInTriangle(pt a, pt b, pt c, pt p) {
    ld s1 = cross(b - a, p - a);
    ld s2 = cross(c - b, p - b);
    ld s3 = cross(a - c, p - c);
    return (sign(s1) >= 0 && sign(s2) >= 0 && sign(s3) >= 0) ||
           (sign(s1) <= 0 && sign(s2) <= 0 && sign(s3) <= 0);
}

// angle abc in radians
ld angle_abc(pt a, pt b, pt c) {
    return acos(clamp<ld>(dot(a - b, c - b) / (abs(a - b) * abs(c - b)), -1, 1));
}

// ============================= Circles ================================

pair<ld, pt> findCircle(pt a, pt b, pt c) {
    pt m1 = (a + b) / 2.0, m2 = (b + c) / 2.0;
    pt ab = b - a, bc = c - b;
    pt center = lineIntersect(m1, m1 + pt(-ab.Y, ab.X),
                                 m2, m2 + pt(-bc.Y, bc.X));
    return {abs(center - a), center};
}

vector<pt> lineCircleIntersect(pt a, pt b, pt center, ld r) {
    pt ab = b - a, ao = center - a;
    pt proj = a + ab * dot(ao, ab) / norm(ab);
    ld d = abs(proj - center);
    if (d > r + EPS) return {};
    if (abs(d - r) < EPS) return {proj};
    ld h = (ld)sqrtl(r*r - d*d);
    pt dir = ab / abs(ab);
    return {proj + dir * h, proj - dir * h};
}

// in 0, 1, 2 pts
vector<pt> circleCircleIntersect(pt c1, ld r1, pt c2, ld r2) {
    ld d = abs(c2 - c1);
    if(d > r1 + r2 + EPS || d < abs(r1 - r2) - EPS) return {};
    if(abs(d) < EPS && abs(r1 - r2) < EPS) return vector(3, c1); // infinity intersection

    ld a = (r1*r1 - r2*r2 + d*d) / (2 * d), h2 = r1*r1 - a*a;
    if (h2 < -EPS) return {};

    pt dir = (c2 - c1) / d, p = c1 + dir * a;
    if (abs(h2) < EPS) return {p};
    ld h = sqrt(h2);
    pt offset = dir * pt(0, 1) * h;
    return {p + offset, p - offset};
}

pair<ld, pt> minimumEnclosingCircle(vector<pt> p) {
    using circle = pair<ld, pt>;
    shuffle(p.begin(), p.end(), mt19937(random_device{}()));
    auto contains = [](circle c, const vector<pt>& pts) {
        return all_of(pts.begin(), pts.end(),
                      [&](auto p) {return abs(p - c.second) <= c.first + EPS;});
    };
    auto circleFrom2 = [](pt a, pt b) {
        pt c = (a + b) / 2.0;
        return circle{abs(a - c), c};
    };
    auto circleFrom3 = [](pt a, pt b, pt c) {
        pt ab = (a + b) / 2.0, ac = (a + c) / 2.0;
        pt ab_perp = (b - a) * pt(0, 1), ac_perp = (c - a) * pt(0, 1);
        pt o = lineIntersect(ab, ab + ab_perp, ac, ac + ac_perp);
        return circle{abs(o - a), o};
    };
    vector<pt> R;
    function<circle(int)> welzl = [&](int n) -> circle {
        if (n == 0 || R.size() == 3) {
            if (R.empty()) return {};
            if (R.size() == 1) return {0, R[0]};
            if (R.size() == 2) return circleFrom2(R[0], R[1]);
            return circleFrom3(R[0], R[1], R[2]);
        }
        pt q = p[n - 1];
        circle D = welzl(n - 1);
        if (contains(D, {q})) return D;
        R.push_back(q);
        auto res = welzl(n - 1);
        R.pop_back();
        return res;
    };
    return welzl((int)p.size());
}

// ===================== polygon ============================

// works for any polygon (returns +1 for ccw, -1 for cw)
ld polygonSign(vector<pt>& p) {
    ld area = 0;
    int n = (int)p.size();
    p.push_back(p[0]);
    for(int i = 0; i < n; ++i) area += cross(p[i], p[i + 1]);
    p.pop_back();
    return sign(0.5 * area);
}

// works for any polygon (removes dups, enforces ccw order)
void normPolygon(vector<pt>& p) {
    vector<pt> res;
    for(auto i : p) if(res.empty() || abs(i - res.back()) > EPS)
            res.push_back(i);

    if(res.size() > 1 && abs(res.front() - res.back()) < EPS)
        res.pop_back();

    if(polygonSign(res) < 0) reverse(res.begin(), res.end());

    p = res;
}

// works for simple polygons with integer coordinates
ll internalPointsCount(vector<pt>& p) {
    ll A2 = 0, B = 0;
    int n = (int)p.size();
    p.push_back(p[0]);
    for (int i = 0; i < n; ++i) {
        pt a = p[i], b = p[i + 1];
        A2 += ll(a.X * b.Y - a.Y * b.X);
        B += __gcd((ll)abs(b.X - a.X), (ll)abs(b.Y - a.Y));
    }
    p.pop_back();
    return (abs(A2) - B + 2) / 2;
}

// works for any polygon (cw or ccw, convex or not)
ld polygonArea(const vector<pt>& p) {
    int n = (int)p.size();
    ld area = 0;
    for (int i = 0; i+1 < n; ++i)
        area += cross(p[i], p[i + 1]);
    area += cross(p.back(), p.front());
    return fabsl(area) / 2.0;
}

// works for any polygon (cw or ccw, convex or not)
bool ptInPolygon(const vector<pt> &p, pt o) {
    int in = 0, n = (int)p.size();
    for (int i = 0; i+1 < n; ++i) {
        pt a = p[i], b = p[i + 1];
        if (onSegment(a, b, o)) return true;
        if (a.Y > o.Y != b.Y > o.Y) {
            ld x = a.X + (b.X - a.X) *
                         (o.Y - a.Y) / (b.Y - a.Y);
            if(x > o.X) in ^= 1;
        }
    }
    {
        pt a = p.back(), b = p.front();
        if (onSegment(a, b, o)) return true;
        if ((a.Y > o.Y) != (b.Y > o.Y)) {
            ld x = a.X + (b.X - a.X) *
                         (o.Y - a.Y) / (b.Y - a.Y);
            if(x > o.X) in ^= 1;
        }
    }
    return in;
}


// work for simple convex polygon
bool ptInConvex(vector<pt> &poly, pt p) {
    int n = int(poly.size());
    if(n == 1) return sign(abs(poly[0] - p)) == 0;
    if(n == 2) return onSegment(poly[0], poly[1], p);

    pt f = poly[0];

    if(sign(cross(poly[1] - f, p - f)) < 0 || 
        sign(cross(poly[n - 1] - f, p - f)) > 0) return false;

    int l = 1, r = n - 1;
    while(r > l + 1) {
        int mid = (l + r) >> 1;
        if(sign(cross(poly[mid] - f, p - f)) > 0) l = mid;
        else r = mid;
    }
    return ptInTriangle(f, poly[l], poly[r], p);
}

// works for any simple polygon (cw or ccw)
pt polygonCentroid(const vector<pt>& p) {
    ld A = 0, c;
    pt C(0, 0);
    int n = (int)p.size();
    pt cur, nxt;
    for (int i = 0; i+1 < n; ++i) {
        cur = p[i], nxt = p[i + 1];
        c = cross(cur, nxt);
        A += c;
        C += (cur + nxt) * c;
    }
    cur = p.back(), nxt = p.front();
    c = cross(cur, nxt);
    A += c;
    C += (cur+nxt) * c;

    A *= 0.5;
    if (abs(A) < EPS) return C;
    return C / (6.0 * A);
}
```