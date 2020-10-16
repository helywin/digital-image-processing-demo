#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main(int argc, const char *argv[])
{
    Mat img1 = imread("/home/helywin/Pictures/flash.jpg", IMREAD_GRAYSCALE);
    if (img1.empty()) {
        std::cerr << "error image" << std::endl;
        return -1;
    }
    namedWindow("img1", WINDOW_AUTOSIZE);
    imshow("image1", img1);
    waitKey(1000);
    Mat padded;
    int m = getOptimalDFTSize(img1.rows);
    int n = getOptimalDFTSize(img1.cols);
    copyMakeBorder(img1, padded, 0, m - img1.rows, 0, n - img1.cols, BORDER_CONSTANT, Scalar::all(0));
    imshow("img2", padded);
    waitKey(1000);
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);

    split(complexI, planes);

    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    magI += Scalar::all(1);
    log(magI, magI);

    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));
    Mat q2(magI, Rect(0, cy, cx, cy));
    Mat q3(magI, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, NORM_MINMAX);
    imshow("image3", magI);
    waitKey(100000);

    return 0;
}
