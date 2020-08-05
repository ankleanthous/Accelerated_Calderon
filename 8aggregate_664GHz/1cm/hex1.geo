e = 0.045180722891566265;   // mesh element size

Point(1)= {-2.898534/4, 12.90468/4, -1.141851/4, e};
Point(2)= {-4.526406/4, 9.649385/4, -2.171615/4, e};
Point(3)= {-5.423097/4, 9.04695/4, -5.796572/4, e};
Point(4)= {-4.691915/4, 11.69981/4, -8.391772/4, e};
Point(5)= {-3.064042/4, 14.95511/4, -7.362009/4, e};
Point(6)= {-2.16734/4, 15.55754/4, -3.737045/4, e};
Point(7)= {8.412789/4, 10.84695/4, -5.571371/4, e};
Point(8)= {7.516099/4, 10.24451/4, -9.196336/4, e};
Point(9)= {5.888227/4, 6.98922/4, -10.2261/4, e};
Point(10)= {5.157021/4, 4.336359/4, -7.630898/4, e};
Point(11)= {6.053735/4, 4.938794/4, -4.005934/4, e};
Point(12)= {7.681607/4, 8.194089/4, -2.97617/4, e};

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
