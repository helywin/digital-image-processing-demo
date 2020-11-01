//
// Created by helywin on 2020/10/29.
//

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <atomic>
#include <cmath>
#include <cassert>
#include "dftshift.hpp"

using namespace cv;
using std::cout;
using std::endl;
using std::cerr;

#define LOW_PASS

int main()
{
    Mat origin = imread("../assets/dark_character.jpg", IMREAD_GRAYSCALE);
    imshow("origin", origin);
    // https://blog.csdn.net/kuweicai/article/details/76473290
    // 获取最佳尺寸
    int r = getOptimalDFTSize(origin.rows);
    int c = getOptimalDFTSize(origin.cols);
    Mat padded;
    copyMakeBorder(origin, padded, 0, r - origin.rows, 0, c - origin.cols, BORDER_CONSTANT, Scalar::all(0));

    // 创建复数矩阵存储数据
    Mat dst1[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat dst2;
    merge(dst1, 2, dst2);
    // 变换
    dft(dst2, dst2);
    dftshift(dst2);

    //实部虚部分开
    split(dst2, dst1);

    const int x_center = r / 2;
    const int y_center = c / 2;
    auto dist_func = [x_center, y_center](int x, int y) -> float {
        double d2 =  (x - x_center) * (x - x_center) + 
               (y - y_center) * (y - y_center);
        return sqrt(d2);
    };
    Mat mag;
    magnitude(dst1[0], dst1[1], mag);
    log(mag, mag);
    normalize(mag, mag, 1, 0, NORM_MINMAX);
    imshow("before magfilter", mag);
    Mat gaussian = Mat_<float>::zeros(r, c);

#ifdef LOW_PASS
    int d0 = 50; //滤波器半径
#else
    int d0 = 50; //滤波器半径
#endif
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
#ifdef LOW_PASS
            gaussian.at<float>(i, j) = 1 / (1 + pow(dist_func(i, j)/d0, 2.5 * 2));
#else
            gaussian.at<float>(i, j) = 1 / (1 + pow(d0/dist_func(i, j), 2.5 * 2));
#endif
        }
    }
    dst1[0] = dst1[0].mul(gaussian);
    dst1[1] = dst1[1].mul(gaussian);
    magnitude(dst1[0], dst1[1], mag);
#ifdef LOW_PASS
    log(mag, mag);
#endif
    normalize(mag, mag, 1, 0, NORM_MINMAX);
    imshow("spectrum after filter", mag);
    merge(dst1, 2, dst2);

    normalize(gaussian, gaussian, 1, 0, NORM_MINMAX);
    imshow("filter", gaussian);

    idft(dst2, dst2);
    split(dst2, dst1);
    Mat after;
    magnitude(dst1[0], dst1[1], after);
    normalize(after, after, 1, 0, NORM_MINMAX);
    imshow("butterworth fitered image", after);
    waitKey();
}
