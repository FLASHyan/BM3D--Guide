#pragma execution_character_set("utf-8")
#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <vector>
#include <time.h>
#include "bm3d.h"
#include "Newton.h"
#include "guidedfilter.h"
#include "Demons.h"
using namespace cv;
using namespace std;
Demons demons;

int main()
{
	//psnr测试
	//Mat src = imread("picture/500t_bm3d_v.bmp", 0);
	//Mat src1 = imread("picture/denoised2-2_pei1.bmp", 0);
	//Mat src2 = imread("picture/50t_v_pei_bm3d.bmp", 0);

	//Mat Src(src.size(), CV_32FC1);
	//Mat Src1(src.size(), CV_32FC1);
	//Mat Src2(src.size(), CV_32FC1);

	//uchar2float(src, Src);
	//uchar2float(src1, Src1);
	//uchar2float(src2, Src2);

	//cout << "psnr for basic estimate:" << cal_psnr(Src, Src1) << endl;
	//cout << "psnr for finnal estimate:" << cal_psnr(Src, Src2) << endl;

	/*                 微分同胚demons配准                */
	//Mat out(src.size(), CV_32FC1);
	//Mat sx;
	//Mat	sy;
	//demons.register_diffeomorphic_demons(src, src1, 1.15, 0.05, 0.05, 200, out, sx, sy);
	//imwrite("picture/denoised2-2_pei1.bmp", out);

	/*****************导向滤波**********************/
	//Mat p = src;
	//int r = 8; // try r=2, 4, or 8
	//double eps = 0.2 * 0.2; // try eps=0.1^2, 0.2^2, 0.4^2

	//eps *= 255 * 255;   // Because the intensity range of our images is [0, 255]
	//Mat q = guidedFilter(src, p, r, eps);
	//imwrite("picture/test_guide1.bmp", q);

	/*******************转BFI**********************/
	//Mat v_lsci_img(src.rows, src.cols, CV_64FC1, Scalar(0));
	//Mat t_mat(src.rows, src.cols, CV_64FC1, Scalar(0));
	//src.convertTo(t_mat, CV_64FC1);
	//Mat dst = Mat::zeros(t_mat.size(), CV_64FC1);
	//manualNormalize(t_mat, dst, 0, 1);

	//for (int i = 0; i < dst.rows; i++)
	//{
	//	double* p111 = dst.ptr<double>(i);
	//	double* p222 = v_lsci_img.ptr<double>(i);
	//	for (int j = 0; j < dst.cols; j++)
	//	{
	//		double x0 = 0.1;
	//		//uchar* p111 = dst.ptr<uchar>(i);

	//		//p111[j] / 255;
	//		if (p111[j] <= 0)
	//		{
	//			//qDebug() << p111[j] << "  ";
	//			p111[j] = 1;
	//		}

	//		p222[j] = findRoot_Newton(p111[j], x0);
	//		//p222[j] = log_k(p222[j] * 10);
	//	}
	//}
	//Mat v_lsci_img_new(dst.rows, dst.cols, CV_8UC1, Scalar(0));
	//Mat v_lsci_img_new_reverse(dst.rows, dst.cols, CV_8UC1, Scalar(0));
	////changeWindows(v_lsci_img, v_lsci_img_new, 500,300);								//调整窗宽窗位
	//changeWindows3(v_lsci_img, v_lsci_img_new, 160, 50);
	//imwrite("picture/denoised3_1.bmp", v_lsci_img_new);

	/*****************  BM3D  ************************/
	//read picture and add noise
	Mat pic = imread("picture/500t_bm3d_v.bmp", 0);
	int sigma = 25;
	if (pic.empty())
	{
		cout << "load image error!";
		return -1;
	}
	//convert data type
	Mat Pic(pic.size(), CV_32FC1);
	Mat Noisy(pic.size(), CV_32FC1);
	Mat Basic(pic.size(), CV_32FC1);
	Mat Denoised(pic.size(), CV_32FC1);

	uchar2float(pic, Pic);
	//addGuassianNoise(sigma, Pic, Noisy);

	//convert type for displaying
	Mat basic(pic.size(), CV_8U);
	Mat noisy(pic.size(), CV_8U);
	noisy = imread("picture/50t_v_pei.bmp", 0);
	uchar2float(noisy, Noisy);
	Mat denoised(pic.size(), CV_8U);
	//float2uchar(Noisy, noisy);
	imshow("origin", pic);
	imshow("noisy", noisy);
	//cout << "psnr for noisy image:" << cal_psnr(Pic, Noisy)<<endl;
	waitKey(10);

	//calculate time used and psnr
	//double start, stop, duration;
	//start = clock();
	runBm3d(sigma, Noisy, Basic, Denoised);//main denoising method
	//stop = clock();
	//duration = double(stop - start) / 1000;
	cout << "psnr for basic estimate:" << cal_psnr(Pic, Basic) << endl;
	cout << "psnr for final denoised:" << cal_psnr(Pic, Denoised) << endl;
	//cout << "time for BM3D:" << duration << " s" << endl;
	float2uchar(Basic, basic);
	float2uchar(Denoised, denoised);
	namedWindow("basic", 1);
	imshow("basic", basic);
	imshow("denoised", denoised);
	imwrite("picture/50t_v_pei_bm3d.bmp", denoised);
	cv::waitKey(0);

	return 0;
}
