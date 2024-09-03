#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <vector>

using namespace cv;
using namespace std;

void addGuassianNoise(const int sigma, const Mat origin, Mat &noisy);
void uchar2float(const Mat tyuchar, Mat &tyfloat);
void float2uchar(const Mat tyfloat, Mat &tyuchar);
float cal_psnr(const Mat x, const Mat y);
int log2(const int N);

int runBm3d(const int sigma, const Mat image_noisy,
	Mat &image_basic, Mat &image_denoised);

void GetAllBlock(const Mat img, const int width, const int height, const int channels,
	const int patchSize, const int step, vector<Mat>&block, vector<int>&row_idx, vector<int>&col_idx);

void tran2d(vector<Mat> &input, int patchsize);

void getSimilarPatch(const vector<Mat> block, vector<Mat>& sim_patch, vector<int>& sim_num,
	int i, int j, int bn_r, int bn_c, int area, int maxNum, int tao);

float cal_distance(const Mat a, const Mat b);

void tran1d(vector<Mat>&input, int patchSize);

void DetectZero(vector<Mat>&input, float threshhold);

float calculate_weight_hd(const vector<Mat>input, int sigma);
float calculate_weight_wien(const vector<Mat>input, int sigma);

void Inver3Dtrans(vector<Mat>&input, int patchSize);

void aggregation(Mat &numerator, Mat &denominator, vector<int>idx_r, vector<int>idx_c, const vector<Mat> input,
	float weight, int patchSize, Mat window);

void gen_wienFilter(vector<Mat>&input, int sigma);

void wienFiltering(vector<Mat>&input, const vector<Mat>wien, int patchSize);

Mat gen_kaiser(int beta, int length);
void wavedec(float *input, int length);
void waverec(float* input, int length, int N);
Mat tvGuidedFilter(const cv::Mat &noisy_patch, const cv::Mat &guided_patch, int patch_size, float sigma, float lambda);
Mat boxfilter(const cv::Mat &I, int r);
void Inver2Dtrans(vector<Mat>& data, int blockSize);