#include"Newton.h"

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