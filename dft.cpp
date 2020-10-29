//
// Created by helywin on 2020/10/27.
//

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <atomic>

using namespace cv;
using std::cout;
using std::endl;
using std::cerr;

int main()
{
#if 0
    Mat origin = imread("../assets/dark_character.jpg", IMREAD_GRAYSCALE);
    Mat dft_real = Mat::zeros(origin.size(), CV_32F);
    Mat dft_comp = Mat::zeros(origin.size(), CV_32F);
    Mat dft_amp = Mat::zeros(origin.size(), CV_32F);
    Mat dft_phs = Mat::zeros(origin.size(), CV_32F);
    int M = origin.rows;
    int N = origin.cols;
    std::mutex mutex;
    int progress{0};
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {   //u
        for (int j = 0; j < N; ++j) {   //v
            float &real = dft_real.at<float>(i, j);
            float &comp = dft_comp.at<float>(i, j);
            for (int ii = 0; ii < M; ++ii) {    //x
                for (int jj = 0; jj < N; ++jj) {    //y
                    double in = i * ii / (double)M + j * jj / (double) N;
                    in *= 2 * M_PI;
                    real += origin.at<uchar>(ii, jj) * cos(in);
                    comp += - origin.at<uchar>(ii, jj) * sin(in);
                }
            }
            dft_amp.at<float>(i, j) = sqrt(real * real + comp * comp);

            mutex.lock();
            progress += 1;
            std::cout << "\rprogress: " << (double) progress * 100 / M / N << "%";
            std::cout.flush();
            mutex.unlock();
        }

    }
//    dft_amp.convertTo(dft_amp, CV_8U);
    log(dft_amp, dft_amp);
    normalize(dft_amp, dft_amp, 1, 0, NORM_MINMAX);

    dft_amp = dft_amp(Rect(0, 0, dft_amp.cols & -2, dft_amp.rows & -2));

    int cx = dft_amp.cols / 2;
    int cy = dft_amp.rows / 2;
    Mat q0(dft_amp(Rect(0, 0, cx, cy)));
    Mat q1(dft_amp(Rect(cx, 0, cx, cy)));
    Mat q2(dft_amp(Rect(0, cy, cx, cy)));
    Mat q3(dft_amp(Rect(cx, cy, cx, cy)));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    imwrite("spectrum_dft.jpg", dft_amp);

    imshow("spectrum", dft_amp);
    waitKey();
#else
    Mat origin = imread("../assets/dark_character.jpg", IMREAD_GRAYSCALE);
    // https://blog.csdn.net/kuweicai/article/details/76473290
    // 获取最佳尺寸
    int r = getOptimalDFTSize(origin.rows);
    int c = getOptimalDFTSize(origin.cols);
    Mat padded;
    copyMakeBorder(origin, padded, 0, r - origin.rows, 0, c - origin.cols, BORDER_CONSTANT, Scalar::all(0));

    // 另一种搬移频谱到中间的办法
    for (int i = 0; i < padded.rows * padded.cols; ++i) {
        padded.at<uchar>(i) *= ((i / padded.cols + i % padded.cols) % 2 ? 1 : -1);
    }

    // 创建复数矩阵存储数据
    Mat dst1[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat dst2;
    merge(dst1, 2, dst2);
    // 变换
    dft(dst2, dst2);


    //实部虚部分开
    split(dst2, dst1);
    magnitude(dst1[0], dst1[1], dst1[0]);
    Mat magnitudeImage = dst1[0];

    // 对数缩放
    magnitudeImage += Scalar::all(1);
    log(magnitudeImage, magnitudeImage);

    magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

//    int cx = magnitudeImage.cols / 2;
//    int cy = magnitudeImage.rows / 2;
//    Mat q0(magnitudeImage(Rect(0, 0, cx, cy)));
//    Mat q1(magnitudeImage(Rect(cx, 0, cx, cy)));
//    Mat q2(magnitudeImage(Rect(0, cy, cx, cy)));
//    Mat q3(magnitudeImage(Rect(cx, cy, cx, cy)));
//
//    Mat tmp;
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);
//    q1.copyTo(tmp);
//    q2.copyTo(q1);
//    tmp.copyTo(q2);

    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
    imshow("spectrum magnitude", magnitudeImage);
    waitKey();
#endif

}