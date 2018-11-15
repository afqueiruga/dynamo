// ###############################
// characteristic length scale
// ###############################
sph_len = 0.125 ;
far_len = 2.5 ;
// ###############################
// dimensions (unit length)
// ###############################
a = 0.35;
r = 1. ;
l = 5. ;
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
    west = newp;    Point(west) = {c[0] - rho, c[1], c[2], cl};
    north = newp;   Point(north) = {c[0], c[1] + rho , c[2], cl};
    south = newp;   Point(south) = {c[0], c[1] -rho, c[2], cl};
    top = newp;     Point(top) = {c[0], c[1], c[2] + rho, cl};
    bottom = newp;  Point(bottom) = {c[0], c[1], c[2] - rho, cl};
    // ###############################
    // lines
    // ###############################
    lineList = {};
    tmp = newreg;   Line(tmp) = {center, top};      lineList[0] = tmp;
    tmp = newreg;   Line(tmp) = {center, bottom};   lineList[1] = tmp;
    tmp = newreg;   Line(tmp) = {center, north};    lineList[2] = tmp;
    tmp = newreg;   Line(tmp) = {center, south};    lineList[3] = tmp;
    tmp = newreg;   Line(tmp) = {center, east};     lineList[4] = tmp;
    tmp = newreg;   Line(tmp) = {center, west};     lineList[5] = tmp;
    // ###############################
    // circles
    // ###############################
    circleList01 = {};
    tmp = newreg; Circle(tmp) = {top, center, east};    circleList01[0] = tmp;
    tmp = newreg; Circle(tmp) = {east, center, bottom}; circleList01[1] = tmp;
    tmp = newreg; Circle(tmp) = {bottom, center, west}; circleList01[2] = tmp;
    tmp = newreg; Circle(tmp) = {west, center, top};    circleList01[3] = tmp;
    circleList02 = {};
    tmp = newreg; Circle(tmp) = {north, center, east};  circleList02[0] = tmp;
    tmp = newreg; Circle(tmp) = {east, center, south};  circleList02[1] = tmp;
    tmp = newreg; Circle(tmp) = {south, center, west};  circleList02[2] = tmp;
    tmp = newreg; Circle(tmp) = {west, center, north};  circleList02[3] = tmp;
    // ###############################
    // surfaces (lines)
    // ###############################
    tmp = newreg; Line Loop(tmp) = {lineList[0], circleList01[0], -lineList[4]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[4], circleList01[1], -lineList[1]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[1], circleList01[2], -lineList[5]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[5], circleList01[3], -lineList[0]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[2], circleList02[0], -lineList[4]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[4], circleList02[1], -lineList[3]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[3], circleList02[2], -lineList[5]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    tmp = newreg; Line Loop(tmp) = {lineList[5], circleList02[3], -lineList[2]};
    ttmp = newreg; Ruled Surface(ttmp) = {tmp};
    // ###############################
    // surfaces (circles)
    // ###############################
    tmp = newreg; Line Loop(tmp) = {circleList01[3], circleList01[0], -circleList02[0], -circleList02[3]};
    upNorth = newreg; Ruled Surface(upNorth) = {tmp};
    tmp = newreg; Line Loop(tmp) = {-circleList02[1], -circleList02[2], -circleList01[3], -circleList01[0]};
    upSouth = newreg; Ruled Surface(upSouth) = {tmp};
    tmp = newreg; Line Loop(tmp) = {circleList02[3], circleList02[0], circleList01[1], circleList01[2]};
    botNorth = newreg; Ruled Surface(botNorth) = {tmp};
    tmp = newreg; Line Loop(tmp) = {circleList02[1], circleList02[2], -circleList01[2], -circleList01[1]};
    botSouth = newreg; Ruled Surface(botSouth) = {tmp};
    // ###############################
    // outer volume
    // ###############################
    surfaceLoop = newreg; Surface Loop(surfaceLoop) = {upNorth, upSouth, botNorth, botSouth};
    surfaceList[] = {upNorth, upSouth, botNorth, botSouth};
Return
// ###############################
// create sphere
// ###############################
c[] = {0.0, 0.0, 0.0}; rho = a * r; cl = sph_len;
Call ConstructSphereSurface;
intSurfaceLoop = surfaceLoop;
intSurfaceList[] = surfaceList[];
intSphereVolume = newreg; Volume(intSphereVolume) = {intSurfaceLoop};
// ###############################
// create sphere
// ###############################
c[] = {0.0, 0.0, 0.0}; rho = r; cl = sph_len;
Call ConstructSphereSurface;
extSurfaceLoop = surfaceLoop;
extSurfaceList[] = surfaceList[];
ShellVolume = newreg; Volume(ShellVolume) = {extSurfaceLoop, intSurfaceLoop};
// ###############################
// create surrounding box
// ###############################
upNW = newp; Point(upNW) = {-l, l, l, far_len};
upNE = newp; Point(upNE) = {l, l, l, far_len};
upSW = newp; Point(upSW) = {-l, -l, l, far_len};
upSE = newp; Point(upSE) = {l, -l, l, far_len};

upBoxLineA = newreg; Line(upBoxLineA) = {upNW, upNE};
upBoxLineB = newreg; Line(upBoxLineB) = {upNE, upSE};
upBoxLineC = newreg; Line(upBoxLineC) = {upSE, upSW};
upBoxLineD = newreg; Line(upBoxLineD) = {upSW, upNW};

upBoxContour = newreg; Line Loop(upBoxContour) = {upBoxLineA, upBoxLineB, upBoxLineC, upBoxLineD};

BoxSurf = {};
upBoxSurf = newreg; Plane Surface(upBoxSurf) = {upBoxContour};
BoxSurf[0] = upBoxSurf;
lowBoxSurf = Translate {0., 0., -2.0 * l} {Duplicata{Surface{upBoxSurf};}};
BoxSurf[1] = lowBoxSurf[0];

boxLineList[] = {upBoxLineA, upBoxLineB, upBoxLineC, upBoxLineD};
For t In {0:3}
	tmp[] = Extrude {0., 0., -2.0 * l} { Line{boxLineList[t]}; };
	BoxSurf[2+t] = tmp[1];
EndFor

boxSurface = newreg; Surface Loop(boxSurface) = BoxSurf[];
hullVolume = newreg; Volume(hullVolume) = {boxSurface, extSurfaceLoop};
Physical Volume(1) = {ShellVolume};
Physical Volume(2) = {intSphereVolume, hullVolume};
Physical Surface(1) = {intSurfaceList[], extSurfaceList[]};
Physical Surface(2) = {BoxSurf[]};
// ###############################
// meshing options
// ###############################
