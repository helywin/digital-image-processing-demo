//
// Created by helywin on 2020/10/20.
//

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using std::cout;
using std::endl;
using std::cerr;

int main()
{

    Mat origin = imread("../assets/dark_character.jpg", IMREAD_GRAYSCALE);
    Mat after_laplace;
    Mat laplace_kernel = Mat::ones(3, 3, CV_32F);
    laplace_kernel.at<float>(0,0) = 0;
    laplace_kernel.at<float>(0,1) = 1;
    laplace_kernel.at<float>(0,2) = 0;
    laplace_kernel.at<float>(1,0) = 1;
    laplace_kernel.at<float>(1,1) = -4;
    laplace_kernel.at<float>(1,2) = 1;
    laplace_kernel.at<float>(2,0) = 0;
    laplace_kernel.at<float>(2,1) = 1;
    laplace_kernel.at<float>(2,2) = 0;
    filter2D(origin, after_laplace, -1, laplace_kernel);
    imshow("origin", origin);
    imshow("laplace", after_laplace);
    waitKey(1e4);
}