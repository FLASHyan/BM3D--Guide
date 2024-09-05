#pragma once
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;
class Demons
{
public:
	void get_gradient(Mat src, Mat &Fx, Mat &Fy);
	void movepixels_2d2(Mat src, Mat &dst, Mat Tx, Mat Ty, int interpolation);
	void composite(Mat& vx, Mat& vy);
	void exp_field(Mat &vx, Mat &vy, Mat &vx_out, Mat &vy_out);

	void imgaussian(Mat src, Mat& dst, float sigma);
	void demons_update(Mat S, Mat M, Mat Sx, Mat Sy, float alpha, Mat& Tx, Mat& Ty);
	void cal_mask(Mat Tx, Mat Ty, Mat &mask);
	double cal_cc_mask(Mat S1, Mat Si, Mat Mask);
	void register_diffeomorphic_demons(Mat F0, Mat M0, float alpha, float sigma_fluid, float sigma_diffusion, int niter, Mat &Mp, Mat &sx, Mat &sy);
};
