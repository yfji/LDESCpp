#include "correlation.h"
#include "fft_functions.h"

cv::Mat gaussianCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel, float sigma) {
	cv::Mat xy = cv::Mat(cv::Size(w, h), CV_32F, cv::Scalar(0));
	// HOG features
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	float xx=0, yy=0;
	float N = h * w;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);   // Procedure do deal with cv::Mat multichannel bug
		y = x2.row(i).reshape(1, h);
		xx +=cv::norm(x)*cv::norm(x) / N;
		yy += cv::norm(y)*cv::norm(y) / N;
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		xy_temp = fftd(xy_temp, true);
		rearrange(xy_temp);	//rearange or not?
		xy_temp.convertTo(xy_temp, CV_32F);
		xy += xy_temp;
	}

	cv::Mat d;
	cv::max(((xx + yy) - 2. * xy) / (w * h * channel), 0, d);

	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
	return fftd(k);
}

cv::Mat linearCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel) {
	cv::Mat xy = cv::Mat(cv::Size(w, h), CV_32FC2, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);;
		y = x2.row(i).reshape(1, h);
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		xy = xy + xy_temp;
	}

	return xy / (h*w*channel);
}

cv::Mat polynomialCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel) {
	cv::Mat xy = cv::Mat(cv::Size(w, h), CV_32F, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);;
		y = x2.row(i).reshape(1, h);
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		//rearrange(caux);	//rearange or not?
		//caux.convertTo(caux, CV_32F);
		xy_temp = fftd(xy_temp, true);
		xy = xy + xy_temp;
	}
	cv::Mat k;
	cv::pow(xy / (h*w*channel) + 1, 9, k);	//polynomal
	k.convertTo(k, CV_32F);
	return fftd(k);
}

cv::Mat phaseCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel) {
	cv::Mat xy = cv::Mat(h, w, CV_32FC2, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	cv::Mat d;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, w);;
		x = x2.row(i).reshape(1, w);
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		//rearrange(caux);
		cv::mulSpectrums(xy_temp, xy_temp, d, 0, true);
		d = d+0.001;
		d = complexDivision(xy_temp, d);
		xy = xy + d;
	}
	return xy;
}