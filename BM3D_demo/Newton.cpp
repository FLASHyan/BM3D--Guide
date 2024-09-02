#include"Newton.h"

double y = 0.95;
double t = 0.6;

/////  牛顿迭代计算血流速度  /////
double eqn(double kk, double x)
{
	double f = pow(kk, 2) - t * (pow(y, 2) * ((exp(-2 * x) - 1 + 2 * x) / (2 * pow(x, 2))) + 4 * y * (1 - y) * (exp(-x) - 1 + x) / pow(x, 2));
	return f;
}

double findRoot_Newton(double kk, double x0)
{
	double x = x0;      // x0是初始猜测的解，对最后的迭代出x的速度有影响
	double eps = 1e-6;  // 当迭代步长小于等于这个值时，认为找到了一个满足精度要求的解
	int maxIterations = 2000;  // 最大迭代次数
	double fx, dfx, dx;

	for (int i = 0; i < maxIterations; i++)
	{
		fx = eqn(kk, x);					// 调用eqn函数，计算当前x值下方程的函数值，并将结果存储在fx变量中
		dfx = (eqn(kk, x + eps) - fx) / eps; 			// 通过近似计算求解方程在x点的导数值。使用函数值的差分来近似计算导数值，eps是一个极小的值，用于控制近似计算的精度
		dx = fx / dfx;					// 计算迭代步长，即x的更新量
		x -= dx;						// 更新x的值

		// 判断迭代步长是否满足精度要求
		if (std::abs(dx) < eps)
		{
			return x;
		}
		//printf("%d", i);
	}
	return -1;
	//如果没有找到根
	//return std::numeric_limits<double>::quiet_NaN();
}

void manualNormalize(const Mat& src, Mat& dst, double minVal, double maxVal) {
	// 找到输入图像的最小和最大值
	const double minPixel = 0;
	const double maxPixel = 255;
	//minMaxLoc(src, &minPixel, &maxPixel);

	// 执行线性归一化
	dst = (src - minPixel) * ((maxVal - minVal) / (maxPixel - minPixel)) + minVal;
}

//伪彩窗宽窗位调整
void changeWindows3(cv::Mat& src, cv::Mat& dst, double WW, double WL)
{
	int i = 0;
	int j = 0;
	for (i = 0; i < src.rows; i++)
		for (j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = cv::saturate_cast<uchar>(float(255) / double(WW) * (src.at<double>(i, j) - (WL - WW / 2)));
		}
}

double log_k(double x)
{ // 求解以1.1为底的对数值
	return log(x) / log(1.1);
}