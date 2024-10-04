#include"Newton.h"
#include <omp.h>
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

void normalizeAndConvertTo8U(cv::Mat& src, cv::Mat& dst) {
	// 计算最小值和最大值
	double minVal, maxVal;
	cv::minMaxLoc(src, &minVal, &maxVal);

	// 检查最小值是否为负值，如果有负值，进行偏移
	if (minVal < 0) {
		src -= minVal;  // 把最小值移动到0
		maxVal -= minVal;  // 相应调整最大值
		minVal = 0;  // 最小值现在为0
	}

	// 缩放并将浮点图像转换为8位图像
	src.convertTo(dst, CV_8U, 255.0 / maxVal);  // 按最大值缩放
}

uint8_t mapTo8Bit(double value) {
	double normalizedValue = value + 100; // 移动范围到 0 到 400
	return static_cast<uint8_t>((normalizedValue / 200.0) * 255); // 映射到 0 到 255
}

cv::Mat convertImageTo8Bit(const cv::Mat& image64) {
	cv::Mat image8bit(image64.size(), CV_8UC1); // 创建8位单通道图像
#pragma omp parallel for
	for (int y = 0; y < image64.rows; ++y) {
		for (int x = 0; x < image64.cols; ++x) {
			image8bit.at<uint8_t>(y, x) = mapTo8Bit(image64.at<double>(y, x));
		}
	}

	return image8bit;
}
