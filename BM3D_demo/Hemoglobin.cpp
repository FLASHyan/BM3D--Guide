#include "Hemoglobin.h"
#include <omp.h>
void Hemoglobin::calculateHemoglobinConcentration(const cv::Mat & redChannel, const cv::Mat & greenChannel, cv::Mat & HbO, cv::Mat & HbR)
{
	// 使用DPF值
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
			int idx = i * cols + j;  //计算一维索引

			// 获取红光和绿光通道的像素强度
			double Rt_red = static_cast<double>(redData[idx]);
			double Rt_green = static_cast<double>(greenData[idx]);

			// 计算 log(R0 / Rt) 对红光和绿光通道
			double logR_red = logR0_red - std::log(Rt_red);
			double logR_green = logR0_green - std::log(Rt_green);

			// 线性系统： log(R_red) = epsilon_HbO_red * ΔHbO + epsilon_HbR_red * ΔHbR
			//            log(R_green) = epsilon_HbO_green * ΔHbO + epsilon_HbR_green * ΔHbR

			// 将线性方程组表示为矩阵形式
			cv::Matx22d A(epsilon_HbO_red * Da_red, epsilon_HbR_red * Da_red,
				epsilon_HbO_green * Da_green, epsilon_HbR_green * Da_green);

			cv::Vec2d logR(logR_red, logR_green);

			// 求解 ΔHbO 和 ΔHbR
			cv::Vec2d delta_Hb = A.inv() * logR;

			// 将结果存储在 HbO 和 HbR 矩阵中
			HbO_data[idx] = delta_Hb[0];
			HbR_data[idx] = delta_Hb[1];
		}
	}
}

double Hemoglobin::calculateBaselineReflectance(const cv::Mat & channel)
{
	// 计算图像的平均像素值
	cv::Scalar meanValue = cv::mean(channel);
	return meanValue[0];  // 返回通道的平均强度
}
