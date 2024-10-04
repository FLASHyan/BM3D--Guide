#include "Hemoglobin.h"
#include <omp.h>
void Hemoglobin::calculateHemoglobinConcentration(const cv::Mat & redChannel, const cv::Mat & greenChannel, cv::Mat & HbO, cv::Mat & HbR)
{
	// ʹ��DPFֵ
	double Da_red = DPF_red;
	double Da_green = DPF_green;

	HbO.create(redChannel.size(), CV_64F);
	HbR.create(redChannel.size(), CV_64F);
	double logR0_red = std::log(R0_red);
	double logR0_green = std::log(R0_green);

	const uchar* redData = redChannel.data;
	const uchar* greenData = greenChannel.data;
	double* HbO_data = HbO.ptr<double>();
	double* HbR_data = HbR.ptr<double>();
	int rows = redChannel.rows;
	int cols = redChannel.cols;
#pragma omp parallel for
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int idx = i * cols + j;  //����һά����

			// ��ȡ�����̹�ͨ��������ǿ��
			double Rt_red = static_cast<double>(redData[idx]);
			double Rt_green = static_cast<double>(greenData[idx]);

			// ���� log(R0 / Rt) �Ժ����̹�ͨ��
			double logR_red = logR0_red - std::log(Rt_red);
			double logR_green = logR0_green - std::log(Rt_green);

			// ����ϵͳ�� log(R_red) = epsilon_HbO_red * ��HbO + epsilon_HbR_red * ��HbR
			//            log(R_green) = epsilon_HbO_green * ��HbO + epsilon_HbR_green * ��HbR

			// �����Է������ʾΪ������ʽ
			cv::Matx22d A(epsilon_HbO_red * Da_red, epsilon_HbR_red * Da_red,
				epsilon_HbO_green * Da_green, epsilon_HbR_green * Da_green);

			cv::Vec2d logR(logR_red, logR_green);

			// ��� ��HbO �� ��HbR
			cv::Vec2d delta_Hb = A.inv() * logR;

			// ������洢�� HbO �� HbR ������
			HbO_data[idx] = delta_Hb[0];
			HbR_data[idx] = delta_Hb[1];
		}
	}
}

double Hemoglobin::calculateBaselineReflectance(const cv::Mat & channel)
{
	// ����ͼ���ƽ������ֵ
	cv::Scalar meanValue = cv::mean(channel);
	return meanValue[0];  // ����ͨ����ƽ��ǿ��
}
