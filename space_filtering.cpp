#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void mean_filter(const Mat in, Mat out, int width, int height)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            out.at<uchar>(i, j) = sum(in(Rect(j - border_left, i - border_top, width, height)))[0]/width/height;
        }
    }
}

void geometry_average_filter(const Mat in, Mat out, int width, int height)
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
            out.at<uchar>(i, j) = v - 1;
        }
    }
}

void harmonic_mean_filter(const Mat in, Mat out, int width, int height)
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
            out.at<uchar>(i, j) = width * height / v - 1;
        }
    }
}

void anti_harmonic_mean_filter(const Mat in, Mat out, int width, int height, int q)
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
            out.at<uchar>(i, j) = v/v1 - 1;
        }
    }
}

void median_filter(const Mat in, Mat out, int width, int height)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            Mat sub = in(Rect(j - border_left, i - border_top, width, height)).clone();
            sub = sub.reshape(0, 1);
            sort(sub, sub, SORT_EVERY_ROW + SORT_ASCENDING);
            out.at<uchar>(i, j) = sub.at<uchar>(width*height/2);
        }
    }
}

void correct_alpha_mean_filter(const Mat in, Mat out, int width, int height, int d)
{
    int border_top = height/ 2;
    int border_left = width / 2;
    for (int i = border_top; i < in.rows - height; ++i) {
        for (int j = border_left; j < in.cols - width; ++j) {
            Mat sub = in(Rect(j - border_left, i - border_top, width, height)).clone();
            sub = sub.reshape(0, 1);
            sort(sub, sub, SORT_EVERY_ROW + SORT_ASCENDING);
            int sum = 0;
            auto sum_elements = [&sum](uchar e){
                sum += e;
            };
            const auto beg = sub.begin<uchar>();
            const auto end = sub.end<uchar>();
            std::for_each(beg+ d/2, end - d/2, sum_elements);
            out.at<uchar>(i, j) = sum / (width * height - d);
        }
    }
}

void adaptive_median_filter(const Mat in, Mat out, int smax)
{
    int border_top = smax / 2;
    int border_left = smax / 2;
    for (int i = border_top; i < in.rows - smax; ++i) {
        for (int j = border_left; j < in.cols - smax; ++j) {
            int v;
            for (int s = 3; s <= smax; s+=2) {
                Mat sub = in(Rect(j - s/2, i - s/2, s, s)).clone();
                sub = sub.reshape(0, 1);
                sort(sub, sub, SORT_EVERY_ROW + SORT_ASCENDING);
                v = sub.at<uchar>(s*s/2);
                if (v > *sub.begin<uchar>() && v < *sub.end<uchar>()) {
                    if (in.at<uchar>(i, j) > *sub.begin<uchar>() && in.at<uchar>(i, j) < *sub.end<uchar>()) {
                        v = in.at<uchar>(i, j);
                    }
                    break;
                }
            }
            out.at<uchar>(i, j) = v;
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
    Mat black = saltpepper_noise < 64;
    Mat white = saltpepper_noise > 211;

    Mat with_noise = origin.clone();
    with_noise.setTo(255, white);
    with_noise.setTo(0, black);
    imshow("with_noise", with_noise);
#endif
    
    // 均值滤波
    Mat after = Mat::zeros(with_noise.size(), CV_8U);
    mean_filter(with_noise, after, 5,5);
    imshow("after mean", after);

    // 几何均值滤波

    geometry_average_filter(with_noise, after, 3, 3);
    imshow("after geometry", after);

    // 谐波平均滤波器
    harmonic_mean_filter(with_noise, after, 3, 3);
    imshow("harmonic mean", after);

    //反谐波平均滤波器
    anti_harmonic_mean_filter(with_noise, after, 3, 3, 1);
    imshow("anti harmonic mean", after);

    //中值滤波
    median_filter(with_noise, after, 3, 3);
    imshow("median", after);

    //修正alpha均值滤波
    correct_alpha_mean_filter(with_noise, after, 3, 3, 6);
    imshow("correct alpha mean", after);

    //自适应中值滤波器
    adaptive_median_filter(with_noise, after, 7);
    imshow("adaptive median", after);

    waitKey();
}
