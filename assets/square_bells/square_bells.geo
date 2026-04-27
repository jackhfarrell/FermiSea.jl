// Generated from square_bells.control
SetFactory("OpenCASCADE");
lc = 0.06; // characteristic length; adjust as desired

Point(1) = {-0.115, -0.5, 0, lc};
Point(2) = {0.115, -0.5, 0, lc};
Point(3) = {0.115, -0.1, 0, lc};
Point(4) = {0.16, -0.1, 0, lc};
Point(5) = {0.16, -0.215, 0, lc};
Point(6) = {0.56, -0.215, 0, lc};
Point(7) = {0.56, 0.255, 0, lc};
Point(8) = {0.16, 0.255, 0, lc};
Point(9) = {0.16, 0.1, 0, lc};
Point(10) = {0.115, 0.1, 0, lc};
Point(11) = {0.115, 0.5, 0, lc};
Point(12) = {-0.115, 0.5, 0, lc};
Point(13) = {-0.115, 0.1, 0, lc};
Point(14) = {-0.16, 0.1, 0, lc};
Point(15) = {-0.16, 0.255, 0, lc};
Point(16) = {-0.56, 0.255, 0, lc};
Point(17) = {-0.56, -0.215, 0, lc};
Point(18) = {-0.16, -0.215, 0, lc};
Point(19) = {-0.16, -0.1, 0, lc};
Point(20) = {-0.115, -0.1, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 17};
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
Plane Surface(1) = {1};

Physical Surface("domain") = {1};
Physical Curve("contact_bottom") = {1};
Physical Curve("contact_top") = {11};
Physical Curve("walls") = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20};
