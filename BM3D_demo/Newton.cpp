#include"Newton.h"
#include <omp.h>
double y = 0.95;
double t = 0.6;

/////  ţ�ٵ�������Ѫ���ٶ�  /////
double eqn(double kk, double x)
{
	double f = pow(kk, 2) - t * (pow(y, 2) * ((exp(-2 * x) - 1 + 2 * x) / (2 * pow(x, 2))) + 4 * y * (1 - y) * (exp(-x) - 1 + x) / pow(x, 2));
	return f;
}

double findRoot_Newton(double kk, double x0)
{
	double x = x0;      // x0�ǳ�ʼ�²�Ľ⣬�����ĵ�����x���ٶ���Ӱ��
	double eps = 1e-6;  // ����������С�ڵ������ֵʱ����Ϊ�ҵ���һ�����㾫��Ҫ��Ľ�
	int maxIterations = 2000;  // ����������
	double fx, dfx, dx;

	for (int i = 0; i < maxIterations; i++)
	{
		fx = eqn(kk, x);					// ����eqn���������㵱ǰxֵ�·��̵ĺ���ֵ����������洢��fx������
		dfx = (eqn(kk, x + eps) - fx) / eps; 			// ͨ�����Ƽ�����ⷽ����x��ĵ���ֵ��ʹ�ú���ֵ�Ĳ�������Ƽ��㵼��ֵ��eps��һ����С��ֵ�����ڿ��ƽ��Ƽ���ľ���
		dx = fx / dfx;					// ���������������x�ĸ�����
		x -= dx;						// ����x��ֵ

		// �жϵ��������Ƿ����㾫��Ҫ��
		if (std::abs(dx) < eps)
		{
			return x;
		}
		//printf("%d", i);
	}
	return -1;
	//���û���ҵ���
	//return std::numeric_limits<double>::quiet_NaN();
}

void manualNormalize(const Mat& src, Mat& dst, double minVal, double maxVal) {
	// �ҵ�����ͼ�����С�����ֵ
	const double minPixel = 0;
	const double maxPixel = 255;
	//minMaxLoc(src, &minPixel, &maxPixel);

	// ִ�����Թ�һ��
	dst = (src - minPixel) * ((maxVal - minVal) / (maxPixel - minPixel)) + minVal;
}

//α�ʴ���λ����
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
{ // �����1.1Ϊ�׵Ķ���ֵ
	return log(x) / log(1.1);
}

void normalizeAndConvertTo8U(cv::Mat& src, cv::Mat& dst) {
	// ������Сֵ�����ֵ
	double minVal, maxVal;
	cv::minMaxLoc(src, &minVal, &maxVal);

	// �����Сֵ�Ƿ�Ϊ��ֵ������и�ֵ������ƫ��
	if (minVal < 0) {
		src -= minVal;  // ����Сֵ�ƶ���0
		maxVal -= minVal;  // ��Ӧ�������ֵ
		minVal = 0;  // ��Сֵ����Ϊ0
	}

	// ���Ų�������ͼ��ת��Ϊ8λͼ��
	src.convertTo(dst, CV_8U, 255.0 / maxVal);  // �����ֵ����
}

uint8_t mapTo8Bit(double value) {
	double normalizedValue = value + 100; // �ƶ���Χ�� 0 �� 400
	return static_cast<uint8_t>((normalizedValue / 200.0) * 255); // ӳ�䵽 0 �� 255
}

cv::Mat convertImageTo8Bit(const cv::Mat& image64) {
	cv::Mat image8bit(image64.size(), CV_8UC1); // ����8λ��ͨ��ͼ��
#pragma omp parallel for
	for (int y = 0; y < image64.rows; ++y) {
		for (int x = 0; x < image64.cols; ++x) {
			image8bit.at<uint8_t>(y, x) = mapTo8Bit(image64.at<double>(y, x));
		}
	}

	return image8bit;
}
