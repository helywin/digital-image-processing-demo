#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "dftshift.hpp"

using namespace cv;


int main()
{
    Mat origin = imread("../assets/gakki.jpg", IMREAD_GRAYSCALE);
    //turbulance

    //int r = getOptimalDFTSize(origin.rows);
    //int c = getOptimalDFTSize(origin.cols);
    //Mat padded;
    //copyMakeBorder(origin, padded, 0, r - origin.rows, 0, c - origin.cols, BORDER_REFLECT, Scalar::all(0));
    imshow("gakki", origin);
    Mat dst1[] = { Mat_<float>(origin), Mat::zeros(origin.size(), CV_32F)};
    Mat dst2;
    merge(dst1, 2, dst2);
    dft(dst2, dst2);
    dftshift(dst2);

    split(dst2, dst1);

    Mat mag;
    magnitude(dst1[0], dst1[1], mag);
    log(mag, mag);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    imshow("magnitude", mag);

    Mat turbulance = Mat::zeros(dst2.size(), CV_32F);
    int r = turbulance.rows;
    int c = turbulance.cols;
    float k = 0.0025;

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float dx = j - c/2.0;
            float dy = i - r/2.0;
            turbulance.at<float>(i,j) = exp(-k * pow(dx*dx + dy*dy, 5/6.0));
        }
    }

    dst1[0] = dst1[0].mul(turbulance);
    dst1[1] = dst1[1].mul(turbulance);
    merge(dst1, 2, dst2);
    idft(dst2, dst2);
    split(dst2, dst1);
    magnitude(dst1[0], dst1[1], mag);
    //log(mag, mag);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    imshow("after turbulance", mag);
    waitKey();
}
