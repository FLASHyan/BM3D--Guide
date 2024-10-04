#pragma once
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <vector>
using namespace cv;
using namespace std;

double eqn(double kk, double x);
double findRoot_Newton(double kk, double x0);
void manualNormalize(const Mat& src, Mat& dst, double minVal, double maxVal);
void changeWindows3(cv::Mat& src, cv::Mat& dst, double WW, double WL);
double log_k(double x);
void normalizeAndConvertTo8U(cv::Mat& src, cv::Mat& dst);
uint8_t mapTo8Bit(double value);
Mat convertImageTo8Bit(const cv::Mat& image64);