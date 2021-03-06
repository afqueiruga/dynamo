// ###############################
// characteristic length scale
// ###############################
sph_len = 0.2 ;
far_len = 5.0 ;
// ###############################
// dimensions (unit length)
// ###############################
r = 1. ;
l = 8. ;
h = 50. ;
// ###############################
// function construction sphere of radius rho
// ###############################
c[] = {0.0, 0.0, 0.0}; rho = 1.0; cl = 0.1;
Macro ConstructSphereSurface
    // ###############################
    // construction points
    // ###############################
    center = newp;  Point(center) = {c[0], c[1], c[2], cl};
    east = newp;    Point(east) = {c[0] + rho, c[1], c[2], cl};
    north = newp;   Point(north) = {c[0], c[1] + rho , c[2], cl};
    south = newp;   Point(south) = {c[0], c[1] -rho, c[2], cl};
    // ###############################
    // lines
    // ###############################
    lineList = {};
    tmp = newreg;   Line(tmp) = {center, north};    lineList[0] = tmp;
    tmp = newreg;   Line(tmp) = {center, south};    lineList[1] = tmp;
    tmp = newreg;   Line(tmp) = {center, east};     lineList[2] = tmp;
    // ###############################
    // circles
    // ###############################
    circleList = {};
    tmp = newreg; Circle(tmp) = {north, center, east};    circleList[0] = tmp;
    tmp = newreg; Circle(tmp) = {east, center, south}; 	  circleList[1] = tmp;
    // ###############################
    // surfaces (lines)
    // ###############################
    tmp = newreg; Line Loop(tmp) = {lineList[0], circleList[0], -lineList[2]};
    tmp = newreg; Line Loop(tmp) = {lineList[2], circleList[1], -lineList[1]};
    // ###############################
    // surfaces (lines)
    // ###############################
    tmp = newreg; Line Loop(tmp) = {lineList[0], circleList[0], -lineList[2]};
    topSurf = newreg; Plane Surface(topSurf) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[2], circleList[1], -lineList[1]};
    botSurf = newreg; Plane Surface(botSurf) = {tmp};
Return
// ###############################
// create sphere
// ###############################
c[] = {0.0, 0.0, 0.0}; rho = r; cl = sph_len;
Call ConstructSphereSurface;

// ###############################
// create surrounding box
// ###############################
boxNorth = newp; Point(boxNorth) = {0.0, h, 0.0, far_len};
boxNorthEast = newp; Point(boxNorthEast) = {l, h, 0.0, far_len};
boxEast = newp; Point(boxEast) = {l, 0.0, 0.0, far_len};
boxSouthEast = newp; Point(boxSouthEast) = {l, -h, 0.0, far_len};
boxSouth = newp; Point(boxSouth) = {0.0, -h, 0.0, far_len};

boxLineA = newreg; Line(boxLineA) = {north, boxNorth};
boxLineB = newreg; Line(boxLineB) = {boxNorth, boxNorthEast};
boxLineC = newreg; Line(boxLineC) = {boxNorthEast, boxEast};
boxLineD = newreg; Line(boxLineD) = {boxEast, east};
boxLineE = newreg; Line(boxLineE) = {boxEast, boxSouthEast};
boxLineF = newreg; Line(boxLineF) = {boxSouthEast, boxSouth};
boxLineG = newreg; Line(boxLineG) = {boxSouth, south};

boxContourNorth = newreg; Line Loop(boxContourNorth) = {boxLineA, boxLineB, boxLineC, boxLineD, -circleList[0]};
boxContourSouth = newreg; Line Loop(boxContourSouth) = {-boxLineD, boxLineE, boxLineF, boxLineG, -circleList[1]};

boxSurf = {};
boxSurfNorth = newreg; Plane Surface(boxSurfNorth) = {boxContourNorth};
boxSurf[0] = boxSurfNorth;
boxSurfSouth = newreg; Plane Surface(boxSurfSouth) = {boxContourSouth};
boxSurf[1] = boxSurfSouth;

// ###############################
// physical surfaces
// ###############################
Physical Surface(1) = {topSurf, botSurf};
Physical Surface(2) = {boxSurf[]};

//+
newsurf[] = Extrude {2, 0, 0} {
  Line{14}; Line{16}; 
};
Physical Surface(3) = {newsurf[]};
newsurff[] = Extrude {50.0, 0, 0} {
  Line{23}; Line{27}; 
};
Physical Surface(4) = {newsurff[]};


// ###############################
// attractors
// ###############################

attrLine = newreg;
Line(tmp) = {north, south};

Field[1] = Attractor;
Field[1].NNodesByEdge = 100;
Field[1].EdgesList = {circleList[0], circleList[1]};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = sph_len;
Field[2].LcMax = far_len;
Field[2].DistMin = 0.2 * r ;
Field[2].DistMax = 5.5 * r;

Background Field = 2;
