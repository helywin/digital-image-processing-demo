#include "dftshift.hpp"
#include <iostream>

using namespace cv;

void dftshift(Mat mat) 
{
    int cx = mat.cols / 2;
    int cy = mat.rows / 2;
    Mat q0(mat(Rect(0,0,cx,cy)));
    Mat q1(mat(Rect(cx,0,cx,cy)));
    Mat q2(mat(Rect(0,cy,cx,cy)));
    Mat q3(mat(Rect(cx,cy,cx,cy)));
    
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
