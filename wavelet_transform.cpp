/**
 * @file wavelet_transform.cpp
 * @brief wavelet_transform
 * @author helywin
 * @version 1.0
 * @date 2020-11-21
 */

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using std::cout;
using std::endl;
using std::cerr;


/**
 * @brief generate WHT mat
 *
 * @param in
 * @param n
 */
void gen_wht_core(Mat &in, int n)
{
    auto join = [](Mat &in_mat) {
        auto size = in_mat.size();
        auto size_after = size * 2;
        Mat out = Mat::zeros(size_after, CV_8S);
        in_mat.copyTo(out(Rect{0, 0, size.width, size.height}));
        in_mat.copyTo(out(Rect{size.width, 0, size.width, size.height}));
        in_mat.copyTo(out(Rect{0, size.height, size.width, size.height}));
        in_mat = -in_mat;
        in_mat.copyTo(out(Rect{size.width, size.height, size.width, size.height}));
        out.copyTo(in_mat);
        out.release();
    };
    in = Mat::ones(2,2,CV_8S);
    //in.at<char>(0,0) = 1;
    //in.at<char>(0,1) = 1;
    //in.at<char>(1,0) = 1;
    in.at<char>(1,1) = -1;
    for (int i = 0; i < n-1; ++i) {
        join(in);
    }
}


/**
 * @brief resort WHT mat
 *
 * @param in
 */
void sort_wht_core(Mat &in)
{
    Mat out = in.clone();
    for (int i = 0; i < in.rows; ++i) {
        Mat row = in.row(i);
        Mat diff = row(Rect{1,0,row.size().width - 1, 1}) - row(Rect{0,0,row.size().width - 1, 1});
        int diff_count = (sum(abs(diff)) / 2)(0);
        row.copyTo(out.row(diff_count));
    }
    out.copyTo(in);
    Mat wht_show = out.clone();
    wht_show += 1;
    wht_show *= 127;
    wht_show.convertTo(wht_show, CV_8U);
    imshow("wht_core", wht_show);
}

/**
 * @brief Walsh-Hadmard Transform (WHT)
 *
 * @param in
 * @param out
 */
void wht(Mat in, Mat &out)
{
    auto mat_size = in.size();
    int max_edge = max(mat_size.width, mat_size.height);
    int n = ceil(log(max_edge)/log(2));
    max_edge = pow(2, n);
    cout << "max_edge: " << max_edge << "n: " << n << std::endl;
    Mat expand;
    copyMakeBorder(in, expand, 0, max_edge - mat_size.height, 0, max_edge - mat_size.width, BORDER_CONSTANT);
    cout << "after copy: " << expand.size() << endl;
    Mat wht_core;
    gen_wht_core(wht_core, n);
    sort_wht_core(wht_core);
    expand.convertTo(expand, CV_32F);
    out = expand.clone();
    wht_core.convertTo(wht_core, CV_32F);
    int count = 0;
    std::mutex count_mutex;
    cout << "begin wht" << endl;
    //waitKey();
#pragma omp parallel for
    for (int i = 0; i < wht_core.rows; ++i) {
        for (int j = 0; j < wht_core.cols; ++j) {
            auto row = wht_core.row(i);
            auto col = wht_core.col(j);
            out.at<float>(i,j) = sum(row * expand * col)(0);
        }
        count_mutex.lock();
        count += 1;
        cout << "\rpercentage: " << count*100.0/wht_core.rows << "%                ";
        cout.flush();
        count_mutex.unlock();
    }
    FileStorage fs("data.yml", FileStorage::WRITE);
    fs << "after" << out;
}

int main()
{
    Mat origin = imread("../assets/gakki.jpg", IMREAD_GRAYSCALE);
    resize(origin, origin, origin.size() / 2);
    imshow("origin", origin);

    Mat after_wht;
    wht(origin, after_wht);
    double min;
    double max;
    minMaxLoc(after_wht, &min, &max);
    after_wht = (after_wht - min) / (max - min) * 256;
    FileStorage fs("data.yml", FileStorage::WRITE);
    fs << "normalize" << after_wht;
    //after_wht *= 256;
    after_wht.convertTo(after_wht, CV_8U);
    imshow("after", after_wht);
    waitKey();
}

