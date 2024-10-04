#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
class Hemoglobin
{
public:
	void calculateHemoglobinConcentration(const cv::Mat& redChannel, const cv::Mat& greenChannel, cv::Mat& HbO, cv::Mat& HbR);
	double calculateBaselineReflectance(const cv::Mat& channel);
private:
	//Ħ������ϵ��
	const double epsilon_HbO_red = 0.034;
	const double epsilon_HbR_red = 0.132;
	const double epsilon_HbO_green = 5.800;
	const double epsilon_HbR_green = 5.400;
	//���߷����ǿ��
	const double R0_red = 85.21;  // 18.4724
	const double R0_green = 43.854;   //10.0443
	// ΢�ֹ�����ӣ����ݲ�����
	const double DPF_red = 0.173;  // ��� (630 nm) �� DPF
	const double DPF_green = 0.050;  // �̹� (530 nm) �� DPF
};
