e = 0.09230769230769231;   // mesh element size

Point(1)= {1.796229/4, -4.457849/4, -0.8796997/4, e};
Point(2)= {2.173801/4, -6.200473/4, -1.100482/4, e};
Point(3)= {1.607539/4, -7.063025/4, -2.571347/4, e};
Point(4)= {0.6637092/4, -6.182931/4, -3.821442/4, e};
Point(5)= {0.2861371/4, -4.440284/4, -3.600662/4, e};
Point(6)= {0.8523972/4, -3.577754/4, -2.129787/4, e};
Point(7)= {5.186406/4, -2.33499/4, -4.527092/4, e};
Point(8)= {4.620142/4, -3.197541/4, -5.997967/4, e};
Point(9)= {4.99773/4, -4.940165/4, -6.218747/4, e};
Point(10)= {5.94156/4, -5.82026/4, -4.968652/4, e};
Point(11)= {6.507801/4, -4.95773/4, -3.497778/4, e};
Point(12)= {6.130236/4, -3.215083/4, -3.276998/4, e};

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
