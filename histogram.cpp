//
// Created by helywin on 2020/10/15.
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
    Mat img1 = origin.clone();
    Mat img2 = origin.clone();
    imwrite("dark_character1.jpg", origin);
    if (origin.empty()) {
        cerr << "open image failed" << endl;
        exit(-1);
    }
//    double maxValue = 0;
//    Point maxLoc;
//    minMaxLoc(origin, nullptr, &maxValue, nullptr, &maxLoc);
//    cout << maxValue << maxLoc << endl;
    auto table = Vec<int, 256>::zeros();
    auto table1 = Vec<int, 256>::zeros();
    auto table2 = Vec<int, 256>::zeros();
    auto tableT = Vec<int, 256>::zeros();
    auto pz = Vec<double, 256>::zeros();
    auto fx = Vec<double, 256>::zeros();
    auto fx1 = Vec<double, 256>::zeros();
    int total = 0;
    for (auto it = origin.begin<uchar>(); it != origin.end<uchar>(); ++it) {
        ++table(*it);
        ++total;
    }
    cout << "total:" << sum(table) << endl;
    cout << table << endl;
    double summer = 0;
    for (int i = 0; i < table.rows; ++i) {
        summer += 255.0 * table(i) / origin.rows / origin.cols;
        fx(i) = summer;
        table1(i) = round(summer);
    }
    cout << table1 << endl;

    for (auto it = img1.begin<uchar>(); it != img1.end<uchar>(); ++it) {
        *it = table1(*it);
    }
    imwrite("dark_character2.jpg", img1);

    // function like a rectangle
    for (int i = 0; i < 256; ++i) {
        if (i < 128) {
            pz(i) = i / 128.0 / 127.0;
        } else {
            pz(i) = -(i - 1) / 128.0 / 127.0 + 1 / 64.0;
        }
    }
    cout << "sum pz" << sum(pz) << endl;

    double summer1 = 0;
    for (int i = 0; i < 256; ++i) {
        summer1 += pz(i) * 255;
        fx1(i) = summer1;
        table2(i) = round(summer1);
    }
    cout << table2 << endl;
    for (int i = 0; i < 256; ++i) {
        const auto v = table1(i);
        bool find = false;
        for (int j = 0; j < 255; ++j) {
            if ((table2(j) - v) * (table2(j + 1) - v) <= 0) {
                if (abs(table2(j) - v) < abs(table2(j + 1) - v)) {
                    tableT(i) = j;
                } else {
                    tableT(i) = j + 1;
                }
                find = true;
                break;
            }
        }
        if (!find) {
            cout << "not find " << endl;
        }
    }
    cout << tableT << endl;
    for (auto it = img2.begin<uchar>(); it != img2.end<uchar>(); ++it) {
        *it = tableT(*it);
    }
    imwrite("dark_character3.jpg", img2);
}