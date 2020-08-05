e = 0.6;   // mesh element size

Point(1)= {8.51234/4 + 0.1, 1.907208/4, 10.88121/4 + 0.1, e};
Point(2)= {11.05626/4 + 0.1, 6.200449/4, 10.41756/4 + 0.1, e};
Point(3)= {15.51033/4 + 0.1, 6.238463/4, 8.120166/4 + 0.1, e};
Point(4)= {17.4205/4 + 0.1, 1.983234/4, 6.286383/4 + 0.1, e};
Point(5)= {14.87657/4 + 0.1, -2.31/4, 6.750023/4 + 0.1, e};
Point(6)= {10.42248/4 + 0.1, -2.348014/4, 9.047423/4 + 0.1, e};
Point(7)= {4.259314/4 + 0.1, 0.0178113/4, -2.862293/4 + 0.1, e};
Point(8)= {8.71338/4 + 0.1, 0.05582505/4, -5.159693/4 + 0.1, e};
Point(9)= {11.2573/4 + 0.1, 4.349054/4, -5.623333/4 + 0.1, e};
Point(10)= {9.347163/4 + 0.1, 8.604278/4, -3.789551/4 + 0.1, e};
Point(11)= {4.893073/4 + 0.1, 8.566265/4, -1.492144/4 + 0.1, e};
Point(12)= {2.349149/4 + 0.1, 4.273026/4, -1.02852/4 + 0.1, e};

Line(1) = {7,8};
Line(2) = {8,9};
Line(3) = {9,10};
Line(4) = {10,11};
Line(5) = {11,12};
Line(6) = {12,7};
Line(7) = {1,12};
Line(8) = {2,11};
Line(9) = {3,10};
Line(10) = {4,9};
Line(11) = {5,8};
Line(12) = {6,7};
Line(13) = {1,6};
Line(14) = {6,5};
Line(15) = {5,4};
Line(16) = {4,3};
Line(17) = {3,2};
Line(18) = {2,1};

Line Loop(1) = {-1, -6, -5, -4, -3, -2};
Line Loop(2) = {1, -11, -14, 12};
Line Loop(3) = {2,-10,-15,11};
Line Loop(4) = {3,-9,-16,10};
Line Loop(5) = {4, -8, -17, 9};
Line Loop(6) = {5, -7, -18, 8};
Line Loop(7) = {6, -12, -13, 7};
Line Loop(8) = {13, 14, 15, 16, 17, 18};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8};