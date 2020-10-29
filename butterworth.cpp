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

using namespace cv;
using std::cout;
using std::endl;
using std::cerr;

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

    int cx = c/2;
    int cy = r/2;

    auto adjust = [cx,cy](Mat dst) {
        Mat q0(dst(Rect(0,0,cx,cy)));
        Mat q1(dst(Rect(cx,0,cx,cy)));
        Mat q2(dst(Rect(0,cy,cx,cy)));
        Mat q3(dst(Rect(cx,cy,cx,cy)));
    
        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    };
    adjust(dst2);

    //实部虚部分开
    split(dst2, dst1);
    //Mat mag, angle;
    //cartToPolar(dst1[0], dst1[1], mag, angle);
    //magnitude(dst1[0], dst1[1], mag);
    //phase(dst1[0], dst1[1], angle);
    //normalize(angle, angle, 0, 1, NORM_MINMAX);
    //imshow("phase", angle);
    //waitKey();

    const int x_center = r / 2;
    const int y_center = c / 2;
    auto dist_func = [x_center, y_center](int x, int y) -> float {
        return (x - x_center) * (x - x_center) + 
               (y - y_center) * (y - y_center);
    };
    Mat mag;
    magnitude(dst1[0], dst1[1], mag);
    log(mag, mag);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    imshow("before magfilter", mag);

    int d0 = 200;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            auto &v1 = dst1[0].at<float>(i, j);
            auto &v2 = dst1[1].at<float>(i, j);
            v1 = v1 / ( 1 + pow(dist_func(i, j)/d0, 2.5));
            v2 = v2 / ( 1 + pow(dist_func(i, j)/d0, 2.5));
        }
    }
    magnitude(dst1[0], dst1[1], mag);
    log(mag, mag);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    imshow("magfilter", mag);
    //polarToCart(mag, angle, dst1[0], dst1[1]);
    merge(dst1, 2, dst2);
    //adjust(dst2);
    split(dst2, dst1);
    magnitude(dst1[0], dst1[1], mag);
    log(mag, mag);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    imshow("magfilter1", mag);

    idft(dst2, dst2);
    split(dst2, dst1);
    Mat after;
    magnitude(dst1[0], dst1[1], after);
    normalize(after, after, 1, 0, NORM_MINMAX);
    //after  = after / r / c;
    //double minv, maxv;
    //minMaxLoc(after, &minv, &maxv, nullptr, nullptr);
    //std::cout << "min: " << minv << "maxv: " << maxv << std::endl;
    //after.convertTo(after, CV_8U);
    //for (int i = 0; i < r * c; ++i) {
    //    after.at<float>(i) *= ((i / c + i % c) % 2 ? -1 : 1);
    //}
    imshow("butterworth fitered image", after);
    waitKey();

}
