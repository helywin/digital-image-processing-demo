#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void mean_filter(Mat in, int width, int height)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            in.at<uchar>(i, j) = sum(in(Rect(j - border_left, i - border_top, width, height)))[0]/width/height;
        }
    }
}

void geometry_average_filter(Mat in, int width, int height)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            double v = 1;
            Mat sub = in(Rect(j - border_left, i - border_top, width, height));
            for (int k = 0; k < sub.rows * sub.cols; ++k) {
                // 需要+1防止出现0导致周围都为0
                v *= pow(sub.at<uchar>(k) + 1, 1.0/ width / height);
            }
            in.at<uchar>(i, j) = v - 1;
        }
    }
}

void harmonic_mean_filter(Mat in, int width, int height)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            double v = 0;
            Mat sub = in(Rect(j - border_left, i - border_top, width, height));
            for (int k = 0; k < sub.rows * sub.cols; ++k) {
                // 需要+1防止出现0导致周围都为0
                v += 1.0/(sub.at<uchar>(k) + 1);
            }
            in.at<uchar>(i, j) = width * height / v - 1;
        }
    }
}

void anti_harmonic_mean_filter(Mat in, int width, int height, int q)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            double v = 0;
            double v1 = 0;
            Mat sub = in(Rect(j - border_left, i - border_top, width, height));
            for (int k = 0; k < sub.rows * sub.cols; ++k) {
                // 需要+1防止出现0导致周围都为0
                v += pow(sub.at<uchar>(k) + 1, q+1);
                v1 += pow(sub.at<uchar>(k) + 1, q);
            }
            in.at<uchar>(i, j) = v/v1 - 1;
        }
    }
}

void median_filter(Mat in, int width, int height)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            double v = 0;
            double v1 = 0;
            Mat sub = in(Rect(j - border_left, i - border_top, width, height));
            sub.reshape(1).copyTo(sub);
            sort(sub, sub, SORT_EVERY_COLUMN + SORT_ASCENDING);
            in.at<uchar>(i, j) = sub.at<uchar>(width * height / 2);
        }
    }
}

int main()
{
    Mat origin = imread("../assets/gakki.jpg", IMREAD_GRAYSCALE);
    imshow("gakki", origin);
    
#ifdef GAUSSIAN
    // 高斯噪声
    Mat with_noise = origin.clone();
    randn(with_noise, 0, 50);
    with_noise += origin;
    imshow("with_noise", with_noise);

#else
    //椒盐噪声
    Mat saltpepper_noise = Mat::zeros(origin.size(), CV_8U);
    randu(saltpepper_noise, 0, 255);
    Mat black = saltpepper_noise < 30;
    Mat white = saltpepper_noise > 225;

    Mat with_noise = origin.clone();
    with_noise.setTo(255, white);
    with_noise.setTo(0, black);
    imshow("with_noise", with_noise);
#endif
    
    // 均值滤波
    Mat after;
    with_noise.copyTo(after);
    mean_filter(after, 5,5);
    imshow("after mean", after);

    // 几何均值滤波

    with_noise.copyTo(after);
    geometry_average_filter(after, 3, 3);
    imshow("after geometry", after);

    // 谐波平均滤波器
    with_noise.copyTo(after);
    harmonic_mean_filter(after, 3, 3);
    imshow("harmonic mean", after);

    //反谐波平均滤波器
    with_noise.copyTo(after);
    anti_harmonic_mean_filter(after, 3, 3, 1);
    imshow("anti harmonic mean", after);

    //中值滤波
    with_noise.copyTo(after);
    median_filter(after, 3, 3);
    imshow("median", after);

    waitKey();
}
