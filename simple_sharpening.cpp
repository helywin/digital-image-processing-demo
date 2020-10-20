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
    laplace_kernel.at<float>(0, 0) = 0;
    laplace_kernel.at<float>(0, 1) = 1;
    laplace_kernel.at<float>(0, 2) = 0;
    laplace_kernel.at<float>(1, 0) = 1;
    laplace_kernel.at<float>(1, 1) = -4;
    laplace_kernel.at<float>(1, 2) = 1;
    laplace_kernel.at<float>(2, 0) = 0;
    laplace_kernel.at<float>(2, 1) = 1;
    laplace_kernel.at<float>(2, 2) = 0;
    filter2D(origin, after_laplace, -1, laplace_kernel);
    after_laplace += 128;

    Mat gauss_kernel = Mat::ones(3, 3, CV_32F);
    gauss_kernel.at<float>(0, 0) = 0.3679;
    gauss_kernel.at<float>(0, 1) = 0.6065;
    gauss_kernel.at<float>(0, 2) = 0.3679;
    gauss_kernel.at<float>(1, 0) = 0.6065;
    gauss_kernel.at<float>(1, 1) = 1.0;
    gauss_kernel.at<float>(1, 2) = 0.6065;
    gauss_kernel.at<float>(2, 0) = 0.3679;
    gauss_kernel.at<float>(2, 1) = 0.6065;
    gauss_kernel.at<float>(2, 2) = 0.3679;
    gauss_kernel /= 4.8976;
    Mat after_gauss;
    filter2D(origin, after_gauss, -1, gauss_kernel);
    Mat sharpening = 2 * origin - after_gauss;

    Mat sobel_y_kernel = Mat::ones(3, 3, CV_32F);
    sobel_y_kernel.at<float>(0, 0) = -1;
    sobel_y_kernel.at<float>(0, 1) = -2;
    sobel_y_kernel.at<float>(0, 2) = -1;
    sobel_y_kernel.at<float>(1, 0) = 0;
    sobel_y_kernel.at<float>(1, 1) = 0;
    sobel_y_kernel.at<float>(1, 2) = 0;
    sobel_y_kernel.at<float>(2, 0) = 1;
    sobel_y_kernel.at<float>(2, 1) = 2;
    sobel_y_kernel.at<float>(2, 2) = 1;
    Mat after_sobel_y;
    filter2D(origin, after_sobel_y, -1, sobel_y_kernel);
    after_sobel_y += 128;

    Mat sobel_x_kernel = Mat::ones(3, 3, CV_32F);
    sobel_x_kernel.at<float>(0, 0) = -1;
    sobel_x_kernel.at<float>(0, 1) = 0;
    sobel_x_kernel.at<float>(0, 2) = 1;
    sobel_x_kernel.at<float>(1, 0) = -2;
    sobel_x_kernel.at<float>(1, 1) = 0;
    sobel_x_kernel.at<float>(1, 2) = 2;
    sobel_x_kernel.at<float>(2, 0) = -1;
    sobel_x_kernel.at<float>(2, 1) = 0;
    sobel_x_kernel.at<float>(2, 2) = 1;
    Mat after_sobel_x;
    filter2D(origin, after_sobel_x, -1, sobel_x_kernel);
    after_sobel_x += 128;

    imshow("origin", origin);
    imshow("laplace", after_laplace);
    imshow("sharpening", sharpening);
    imshow("sobel_y", after_sobel_y);
    imshow("sobel_x", after_sobel_x);
    imshow("sobel", (abs(after_sobel_x) + abs(after_sobel_y)) / 2);

    waitKey(1e6);
}