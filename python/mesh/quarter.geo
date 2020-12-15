lr = 1;
Point(1) = {0, 0, 0, lr};
Point(2) = {10, 0, 0, lr};
Line(1) = {1, 2};

Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
Line{1};Layers{8};
}
Extrude {{0, -1, 0}, {0, 0, 0}, Pi/2} {
Surface{4};Layers{8};Recombine;
}
