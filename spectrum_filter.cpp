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

//#define LOW_PASS
//#define HIGH_PASS
//#define LAPLACE
#define HOMOMORPHIC
void show_range(Mat mat, const char *name)
{
    double min, max;
    minMaxLoc(mat, &min, &max);
    std::cout << name << ": min, max " << min << ", " << max << std::endl;
}

int main()
{
    Mat origin = imread("../assets/dark_character.jpg", IMREAD_GRAYSCALE);
    imshow("origin", origin);
    // https://blog.csdn.net/kuweicai/article/details/76473290
    // 获取最佳尺寸
    int r = getOptimalDFTSize(origin.rows);
    int c = getOptimalDFTSize(origin.cols);
    Mat padded;
#ifndef HOMOMORPHIC
    copyMakeBorder(origin, padded, 0, r - origin.rows, 0, c - origin.cols, BORDER_CONSTANT, Scalar::all(0));
#else
    copyMakeBorder(origin, padded, 0, r - origin.rows, 0, c - origin.cols, BORDER_REFLECT, Scalar::all(1));
    padded += 1;
    imshow("padded", padded);
#endif

    // 创建复数矩阵存储数据
    Mat dst1[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat dst2;
#if defined(HOMOMORPHIC)
    log(dst1[0], dst1[0]);
    show_range(dst1[0], "dst1[0]");
#endif
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
//#if defined(HOMOMORPHIC)
    //double min;
    //double max;
    //minMaxLoc(mag, &min, &max);
    //mag = (mag - min) / (max - min) * 256;
    //std::cout << "min, max" << min << ", " << max << std::endl;
//#else
    log(mag, mag);
//#endif

    normalize(mag, mag, 1, 0, NORM_MINMAX);
    imshow("mag before filter", mag);
    Mat filter = Mat_<float>::zeros(r, c);


#if defined(LOW_PASS)
    int d0 = 50; //滤波器半径
#elif defined(HIGH_PASS)
    int d0 = 50; //滤波器半径
#elif defined(LAPLACE)
    double rat = 20;
#elif defined(HOMOMORPHIC)
    double gama_h = 1.5;
    double gama_l = 0.5;
    double homo_c = 1;
    int d0 = 50;
#endif
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            auto &v = filter.at<float>(i, j);
#if defined(LOW_PASS)
            v = 1 / (1 + pow(dist_func(i, j)/d0, 2.5 * 2));
#elif defined(HIGH_PASS)
            v = 1 / (1 + pow(d0/dist_func(i, j), 2.5 * 2));
#elif defined(LAPLACE)
            v = 1 + pow(dist_func(i, j), 2) / r / c * rat;
#elif defined(HOMOMORPHIC)
            v = (gama_h - gama_l) * (1 - exp(-homo_c * pow(dist_func(i, j)/d0, 2))) + gama_l;
#endif
        }
    }
    dst1[0] = dst1[0].mul(filter);
    dst1[1] = dst1[1].mul(filter);
    magnitude(dst1[0], dst1[1], mag);
#if defined(LOW_PASS) || defined(LAPLACE) || defined(HOMOMORPHIC)
    log(mag, mag);
#endif
    normalize(mag, mag, 1, 0, NORM_MINMAX);
    imshow("spectrum after filter", mag);
    merge(dst1, 2, dst2);

    normalize(filter, filter, 1, 0, NORM_MINMAX);
    imshow("filter", filter);

    idft(dst2, dst2);
    split(dst2, dst1);
//#if defined(HOMOMORPHIC)
    //exp(dst1[0], dst1[0]);
    //exp(dst1[1], dst1[1]);
//#endif
    Mat after;
    magnitude(dst1[0], dst1[1], after);
    show_range(after, "after");
#if defined(HOMOMORPHIC)
    double min, max;
    minMaxLoc(after, &min, &max);
    after = (after - min) / (max - min) * 256;
#endif
    normalize(after, after, 1, 0, NORM_MINMAX);
    imshow("fitered image", after);
    waitKey();
}
