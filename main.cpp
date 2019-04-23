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
cv::Rect get_groundtruth(string& line);

int main()
{
	testKCF();
	return 0;
}

cv::Rect get_groundtruth(string& line) {
	std::stringstream ss;
	ss << line;
	string s;
	vector<int> v;
	while (std::getline(ss, s, ',')) {
		v.push_back(atoi(s.c_str()));
	}
	return cv::Rect(v[0], v[1], v[2], v[3]);
}

void testKCF() {
	string img_file = "D:/Dataset/OTB100/Mhyang.txt";
	string label_file = "D:/Dataset/OTB100/Mhyang_label.txt";
	LDESTracker tracker;

	ifstream fin, lfin;
	fin.open(img_file.c_str(), ios::in);
	lfin.open(label_file.c_str(), ios::in);

	int frameIndex = 0;
	cv::Rect new_pos;
	while (!fin.eof()) {
		string filename, gt;
		getline(fin, filename);
		getline(lfin, gt);

		cv::Mat image = cv::imread(filename);
		cv::Rect position = get_groundtruth(gt);
		if (frameIndex == 0){
			new_pos = tracker.testKCFTracker(image, position, true);
			new_pos = position;
		}
		else {
			new_pos = tracker.testKCFTracker(image, new_pos, false);
		}
		++frameIndex;

		cv::rectangle(image, new_pos, cv::Scalar(0, 0, 255), 2);
		cv::imshow("trackKCF", image);
		if (cv::waitKey() == 27)
			break;
	}
}

void testLDES() {
	string img_file = "D:/Dataset/OTB100/Mhyang.txt";
	string label_file = "D:/Dataset/OTB100/Mhyang_label.txt";
	LDESTracker tracker;

	ifstream fin, lfin;
	fin.open(img_file.c_str(), ios::in);
	lfin.open(label_file.c_str(), ios::in);

	int frameIndex = 0;
	cv::Rect new_pos;
	while (!fin.eof()) {
		string filename, gt;
		getline(fin, filename);
		getline(lfin, gt);

		cv::Mat image = cv::imread(filename);
		cv::Rect position = get_groundtruth(gt);
		if (frameIndex == 0) {
			tracker.init(position, image);
			new_pos = position;
		}
		else {
			tracker.update(image);
			new_pos = tracker.cur_position;
		}
		++frameIndex;

		cv::rectangle(image, new_pos, cv::Scalar(0, 0, 255), 2);
		cv::imshow("trackKCF", image);
		if (cv::waitKey(1) == 27)
			break;
	}
}