// LDES.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include "ldes_tracker.h"
using namespace std;

void testKCF();
void testLDES();
void testPhaseCorrelation();

cv::Rect get_groundtruth(string& line, int& start);
std::vector<cv::Point2i> get_rotated_rect(cv::Rect& rec, float degree);

int main()
{
	//testPhaseCorrelation();
	testLDES();
	//testKCF();
	return 0;
}

cv::Rect get_groundtruth(string& line, int& start) {
	std::stringstream ss;
	ss << line;
	string s;
	vector<int> v;
	while (std::getline(ss, s, ',')) {
		v.push_back(atoi(s.c_str()));
	}
	if (v.size() == 5) {
		if(start==-1)
			start = (int)v[0];
		return cv::Rect(v[1], v[2], v[3], v[4]);
	}
	else
		return cv::Rect(v[0], v[1], v[2], v[3]);
}

std::vector<cv::Point2i> get_rotated_rect(cv::Rect& rec, float degree) {
	std::vector<cv::Point2i> rot_rect;
	cv::Point2i center(rec.x + rec.width / 2, rec.y + rec.height / 2);
	cv::Mat M = cv::getRotationMatrix2D(center, degree, 1);
	M.convertTo(M, CV_32F);
	float corners_ptr[12] = {
		rec.x, rec.y, 1.0,\
		rec.x, rec.y + rec.height - 1,1.0,\
		rec.x + rec.width - 1, rec.y + rec.height - 1,1.0,\
		rec.x + rec.width - 1, rec.y,1.0
	};
	cv::Mat corners(4, 3, CV_32F, corners_ptr);
	cv::transpose(M, M);
	cv::Mat wcorners = corners * M;	//4*2
	float *data = (float*)wcorners.data;
	for (int i = 0; i < 4; ++i) {
		cv::Point2i p(data[2 * i], data[2 * i + 1]);
		rot_rect.push_back(p);
	}
	return rot_rect;
}

void testKCF() {
	string img_file = "J:/Dataset/OTB100/Mhyang.txt";
	string label_file = "J:/Dataset/OTB100/Mhyang_label.txt";
	LDESTracker tracker;

	ifstream fin, lfin;
	fin.open(img_file.c_str(), ios::in);
	lfin.open(label_file.c_str(), ios::in);

	int frameIndex = 0;
	cv::Rect new_pos;
	int start = -1;
	while (!fin.eof()) {
		string filename, gt;
		getline(fin, filename);
		getline(lfin, gt);
		if (gt.length() == 0)
			continue;

		cv::Mat image = cv::imread(filename);
		cv::Rect position = get_groundtruth(gt, start);
		if (frameIndex == 0){
			new_pos = tracker.testKCFTracker(image, position, true);
		}
		else {
			new_pos = tracker.testKCFTracker(image, position, false);
		}
		++frameIndex;
		cv::rectangle(image, new_pos, cv::Scalar(0, 255,0), 2);		
		cv::imshow("trackKCF", image);
		if (cv::waitKey(1) == 27)
			break;
	}
}

void testLDES() {
	string img_file = "J:/Dataset/OTB100/Skiing.txt";
	string label_file = "J:/Dataset/OTB100/Skiing_label.txt";
	//string img_file = "J:/Dataset/tracking-traffic/annotations_otb/avi_0.txt";
	//string label_file = "J:/Dataset/tracking-traffic/annotations_otb/avi_0/target_2.txt";
	LDESTracker tracker;

	ifstream fin, lfin;
	fin.open(img_file.c_str(), ios::in);
	lfin.open(label_file.c_str(), ios::in);

	int frameIndex = 0;
	cv::Rect new_pos;
	float rot_degree = 0;
	int start = -1;
	while (!lfin.eof()) {
		string filename, gt;
		getline(fin, filename);
		getline(lfin, gt);

		cv::Mat image = cv::imread(filename);
		cv::Rect position = get_groundtruth(gt, start);
		if (start >= 1 && frameIndex < start - 1) {
			continue;
		}
		if (frameIndex == 0) {
			tracker.init(position, image);
			new_pos = position;
		}
		else {
			new_pos = tracker.update(image);
			rot_degree = tracker.cur_rot_degree;
		}
		++frameIndex;
		auto rot_rect = get_rotated_rect(new_pos, rot_degree);

		cv::rectangle(image, new_pos, cv::Scalar(0, 0, 255), 2);
		cv::circle(image, tracker.cur_pos, 3, cv::Scalar(0, 255, 0), -1);
		for (int i = 0; i < 4; ++i) {
			int  from = i;
			int to = (i + 1) % 4;
			cv::line(image, rot_rect[from], rot_rect[to], cv::Scalar(0, 255, 0), 1);
		}
		cv::imshow("trackLDES", image);
		if (cv::waitKey(1) == 27)
			break;
	}
}

cv::Mat getHistFeatures(cv::Mat& img, int* size) {
	cv::Mat features(img.channels(), img.cols*img.rows, CV_32F);
	vector<cv::Mat > planes(3);
	cv::split(img, planes);
	planes[0].reshape(1, 1).copyTo(features.row(0));
	planes[1].reshape(1, 1).copyTo(features.row(1));
	planes[2].reshape(1, 1).copyTo(features.row(2));
	size[0] = img.rows;
	size[1] = img.cols;
	size[2] = img.channels();
	return features;
}

void testPhaseCorrelation() {
	cv::Rect roi(84, 53, 62, 70);
	LDESTracker tracker;
	float padding = 2.5;
	int sz = 120;

	int window_sz = static_cast<int>(sqrt(roi.area())*padding);
	int cx = (int)(roi.x + roi.width*0.5);
	int cy = (int)(roi.y + roi.height*0.5);

	cv::Rect window(cx - window_sz / 2, cy - window_sz / 2, window_sz, window_sz);

	string img1_path = "./0001.jpg";

	cv::Mat image = cv::imread(img1_path);
	cv::Mat img1 = image(window).clone();
	cv::Mat img2;
	cv::resize(img1, img1, cv::Size(sz, sz));

	float last_scale = 1.1;
	float cur_scale = 1.2;
	cv::Mat rot_matrix = cv::getRotationMatrix2D(cv::Point2f(cx, cy), 20.5, cur_scale);
	rot_matrix.convertTo(rot_matrix, CV_32F);
	rot_matrix.at<float>(0, 2) += window_sz * last_scale * 0.5 - cx;
	rot_matrix.at<float>(1, 2) += window_sz * last_scale * 0.5 - cy;

	cv::warpAffine(image, img2, rot_matrix, cv::Size(window_sz*last_scale,window_sz*last_scale));
	cv::resize(img2, img2, cv::Size(sz, sz));	//cannot resize
	//cv::blur(img2, img2, cv::Size(5, 5));
	//cv::Mat mask(20, 60, CV_8UC3, cv::Scalar(0));
	//mask.copyTo(img2(cv::Rect(10, 40, mask.cols, mask.rows)));
	cv::imshow("img1", img1);
	cv::imshow("img2", img2);

	cv::Mat log1, log2;

	//float mag= sz / (log(sqrt((sz*sz + sz*sz)*0.25)));
	float mag = 30;
	cv::logPolar(img1, log1, cv::Point2f(0.5*sz, 0.5*sz), mag, cv::INTER_LINEAR);
	cv::logPolar(img2, log2, cv::Point2f(0.5*sz, 0.5*sz), mag, cv::INTER_LINEAR);

	int size[3] = { 0 };

	//better with larger featuremap
	bool _hog = false;

	cv::Mat x1, x2;
	if (_hog) {
		cv::Mat _empty;
		x1 = tracker.getFeatures(log1, _empty, size, false);
		x2 = tracker.getFeatures(log2, _empty, size, false);
	}
	else {
		x1 = getHistFeatures(log1, size);
		x2 = getHistFeatures(log2, size);
	}
	cv::Mat rf=phaseCorrelation(x2, x1, size[0], size[1], size[2]);
	cv::Mat res = fftd(rf, true);
	rearrange(res);
	
	cv::Rect center(5, 5, size[1] - 10, size[0] - 10);
	res = res(center).clone();
	cv::Point2i pi;
	
	double pv;
	cv::minMaxLoc(res, NULL, &pv, NULL, &pi);

	pi.x += 5;
	pi.y += 5;

	float px = pi.x, py = pi.y;
	if (px > 0 && px < res.cols - 1) {
		px += tracker.subPixelPeak(res.at<float>(py, px - 1), pv, res.at<float>(py, px + 1));
	}

	if (py > 0 && py < res.rows - 1) {
		py += tracker.subPixelPeak(res.at<float>(py - 1, px), pv, res.at<float>(py + 1, px));
	}

	px -= size[1] * 0.5;
	py -= size[0] * 0.5;
	float rot = -py* 180.0 / (size[0] * 0.5);
	float scale = exp((px)/mag);

	cout <<"rot: " << rot << ", scale: " << 1.0*last_scale*scale << endl;
	cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);
	cv::imshow("log1", log1);
	cv::imshow("log2", log2);
	cv::imshow("res", res);
	cv::waitKey();
}