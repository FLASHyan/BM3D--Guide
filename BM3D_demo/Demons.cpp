#include "Demons.h"

//计算梯度
void Demons::get_gradient(Mat src, Mat & Fx, Mat & Fy)
{
	Mat src_board;
	//边缘扩充
	copyMakeBorder(src, src_board, 1, 1, 1, 1, BORDER_REPLICATE);

	Fx = Mat::zeros(src.size(), CV_32FC1);
	Fy = Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < src.rows; i++)
	{
		float *p_Fx = Fx.ptr<float>(i);
		float *p_Fy = Fy.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			p_Fx[j] = (src_board.ptr<float>(i + 1)[j + 2] - src_board.ptr<float>(i + 1)[j]) / 2.0;
			p_Fy[j] = (src_board.ptr<float>(i + 2)[j + 1] - src_board.ptr<float>(i)[j + 1]) / 2.0;
		}
	}
}

//像素重采样
void Demons::movepixels_2d2(Mat src, Mat & dst, Mat Tx, Mat Ty, int interpolation)
{
	Mat Tx_map(src.size(), CV_32FC1, 0.0);
	Mat Ty_map(src.size(), CV_32FC1, 0.0);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			Tx_map.at<float>(i, j) = j + Tx.at<float>(i, j);
			Ty_map.at<float>(i, j) = i + Ty.at<float>(i, j);
		}
	}

	remap(src, dst, Tx_map, Ty_map, interpolation);
}

void Demons::composite(Mat & vx, Mat & vy)
{
	Mat bxp, byp;
	movepixels_2d2(vx, bxp, vx, vy, INTER_LINEAR);
	movepixels_2d2(vy, byp, vx, vy, INTER_LINEAR);

	vx = vx + bxp;
	vy = vy + byp;
}

//微分同胚映射转换
void Demons::exp_field(Mat & vx, Mat & vy, Mat & vx_out, Mat & vy_out)
{
	Mat normv2 = vx.mul(vx) + vy.mul(vy);

	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(normv2, &minv, &maxv, &pt_min, &pt_max);   //求最大最小值

	float m = sqrt(maxv);
	float n = ceil(log2(m / 0.5));
	n = n > 0.0 ? n : 0.0;

	float a = pow(2.0, -n);

	vx_out = vx * a;
	vy_out = vy * a;
	//n次复合运算
	for (int i = 0; i < (int)n; i++)
	{
		composite(vx_out, vy_out);
	}
}

void Demons::imgaussian(Mat src, Mat & dst, float sigma)
{
	int radius = (int)ceil(3 * sigma);
	int ksize = 2 * radius + 1;

	GaussianBlur(src, dst, Size(ksize, ksize), sigma);
}

//Active Demons位移场
void Demons::demons_update(Mat S, Mat M, Mat Sx, Mat Sy, float alpha, Mat & Tx, Mat & Ty)
{
	Mat diff = S - M;
	Tx = Mat::zeros(S.size(), CV_32FC1);
	Ty = Mat::zeros(S.size(), CV_32FC1);

	Mat Mx, My;
	get_gradient(M, Mx, My);   //求M的梯度

	float alpha_2 = alpha * alpha;

	for (int i = 0; i < S.rows; i++)
	{
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		float* p_diff = diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			float alpha_diff = alpha_2 * p_diff[j] * p_diff[j];
			float a1 = p_sx[j] * p_sx[j] + p_sy[j] * p_sy[j] + alpha_diff;  //分母
			float a2 = p_mx[j] * p_mx[j] + p_my[j] * p_my[j] + alpha_diff;

			if (a1 > 0.00001 && a2 > 0.00001)
			{
				p_tx[j] = p_diff[j] * (p_sx[j] / a1 + p_mx[j] / a2);
				p_ty[j] = p_diff[j] * (p_sy[j] / a1 + p_my[j] / a2);
			}
		}
	}
}

//计算相似度
//获取黑色边缘的mask矩阵
void Demons::cal_mask(Mat Tx, Mat Ty, Mat & mask)
{
	mask.create(Tx.size(), CV_8UC1);

	for (int i = 0; i < Tx.rows; i++)
	{
		float *pTx = Tx.ptr<float>(i);
		float *pTy = Ty.ptr<float>(i);
		uchar *pm = mask.ptr<uchar>(i);
		for (int j = 0; j < Tx.cols; j++)
		{
			int x = (int)(j + pTx[j] + 0.5);
			int y = (int)(i + pTy[j] + 0.5);

			if (x < 0 || x >= Tx.cols || y < 0 || y >= Tx.rows)
			{
				pm[j] = 0;
			}
			else
			{
				pm[j] = 255;
			}
		}
	}
}

//计算归一化互相关系数
double Demons::cal_cc_mask(Mat S1, Mat Si, Mat Mask)
{
	double result;
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;

	for (int i = 0; i < S1.rows; i++)
	{
		for (int j = 0; j < S1.cols; j++)
		{
			if (Mask.ptr<uchar>(i)[j])
			{
				sum1 += (double)(S1.at<float>(i, j)*Si.at<float>(i, j));
				sum2 += (double)(S1.at<float>(i, j)*S1.at<float>(i, j));
				sum3 += (double)(Si.at<float>(i, j)*Si.at<float>(i, j));
			}
		}
	}

	result = sum1 / sqrt(sum2*sum3);   //范围0~1

	return result;
}

//微分同胚demons最终实现
void Demons::register_diffeomorphic_demons(Mat F0, Mat M0, float alpha, float sigma_fluid, float sigma_diffusion, int niter, Mat & Mp, Mat & sx, Mat & sy)
{
	Mat F, M;
	F0.convertTo(F, CV_32F);  //将参考图像和浮动图像都转换为浮点型矩阵
	M0.convertTo(M, CV_32F);

	//初始化位移场为0位移
	Mat vx = Mat::zeros(F.size(), CV_32FC1);
	Mat vy = Mat::zeros(F.size(), CV_32FC1);

	float e_min = -9999999999.9;
	Mat sx_min, sy_min;

	Mat Sx, Sy;
	get_gradient(F, Sx, Sy);  //计算参考图像梯度图

	Mat M1 = M.clone();

	for (int i = 0; i < niter; i++)
	{
		Mat ux, uy;
		//计算Active demons的位移场
		demons_update(F, M1, Sx, Sy, alpha, ux, uy);

		imgaussian(ux, ux, sigma_fluid);  //高斯滤波
		imgaussian(uy, uy, sigma_fluid);  //高斯滤波

		vx = vx + ux;  //将位移场累加
		vy = vy + uy;

		imgaussian(vx, vx, sigma_diffusion);   //高斯滤波
		imgaussian(vy, vy, sigma_diffusion);   //高斯滤波

		exp_field(vx, vy, sx, sy);  //将累加的位移场转换为微分同胚映射

		Mat mask;
		cal_mask(sx, sy, mask);   //计算黑色边缘的mask掩码矩阵
		movepixels_2d2(M, M1, sx, sy, INTER_LINEAR);  //对M像素重采样
		float e = cal_cc_mask(F, M1, mask);  //计算F、M1的相似度

		if (e > e_min)  //如果相似度提高，则更新最佳位移场
		{
			printf("i=%d, e=%f\n", i, e);
			e_min = e;
			sx_min = sx.clone();
			sy_min = sy.clone();
		}
	}

	sx = sx_min.clone();   //得到最优微分同胚映射
	sy = sy_min.clone();

	movepixels_2d2(M, Mp, sx, sy, INTER_LINEAR);
	Mp.convertTo(Mp, CV_8U);
}
