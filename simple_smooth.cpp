//
// Created by helywin on 2020/10/19.
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
//    std::cout << "mat type: " << origin.type();
//    Mat expand;
    Mat after_box, after_gauss;
    int size = 3;
//    int border = size / 2;
//    copyMakeBorder(origin, expand, border, border, border, border, BORDER_CONSTANT);
//    expand.convertTo(expand, CV_32F);
    const Mat box_kernel = Mat::ones(size, size, CV_32F) / size / size;
    filter2D(origin, after_box, -1, box_kernel);

    Mat guass_kernel = Mat::ones(size, size, CV_32F);
    guass_kernel.at<float>(0,0) = 0.3679;
    guass_kernel.at<float>(0,1) = 0.6065;
    guass_kernel.at<float>(0,2) = 0.3679;
    guass_kernel.at<float>(1,0) = 0.6065;
    guass_kernel.at<float>(1,1) = 1.0;
    guass_kernel.at<float>(1,2) = 0.6065;
    guass_kernel.at<float>(2,0) = 0.3679;
    guass_kernel.at<float>(2,1) = 0.6065;
    guass_kernel.at<float>(2,2) = 0.3679;
    guass_kernel /= 4.8976;
    filter2D(origin, after_gauss, -1, guass_kernel);

    after_box.convertTo(after_box, CV_8U);
    after_gauss.convertTo(after_gauss, CV_8U);

    imshow("origin", origin);
    imshow("box", after_box);
    imshow("gauss", after_gauss);
    waitKey(1e4);
}