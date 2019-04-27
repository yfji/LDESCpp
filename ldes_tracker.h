#pragma once
#include <opencv2/opencv.hpp>
#include "fft_functions.h"
#include "correlation.h"
#include "fhog.hpp"
#include "hann.h"

class LDESTracker {
public:
	LDESTracker();
	~LDESTracker();

	void init(const cv::Rect &roi, cv::Mat image);
	void update(cv::Mat image);	//update BGD

	float interp_n;
	float interp_factor; // linear interpolation factor for adaptation
	float sigma; // gaussian kernel bandwidth
	float lambda; // regularization
	int cell_size; // HOG cell size
	int cell_size_search;
	int cell_sizeQ; // cell size^2, to avoid repeated operations
	float padding; // extra area surrounding the target
	float inter_patch_rate;
	float color_update_rate;
	float color_bins;
	float merge_factor;
	float output_sigma_factor; // bandwidth of gaussian target
	int template_sz; // template size

	float scale_step; // scale step for multi-scale estimation
	float scale_weight;  // to downweight detection scores of other scales for added stability

	float train_interp_factor;
	float interp_factor_scale;

	float cscore;
	float sscore;

	cv::Size target_sz;
	cv::Size target_sz0;
	cv::Size window_sz;
	cv::Size window_sz0;

	cv::Size window_sz_search;
	cv::Size window_sz_search0;

	cv::Size feature_sz;
	cv::Size feature_sz0;
	cv::Size feature_size_search;

	cv::Size scale_sz;
	cv::Size scale_sz0;
	cv::Size scale_sz_window;

	cv::Mat hann;
	cv::Mat hann_search;
	cv::Mat hann_scale;

	cv::Mat patch;
	cv::Mat patchL;

	cv::Mat resLocation;
	cv::Mat resScale;

	float peak_val_location;
	float peak_val_scale;

	bool _resize;
	cv::Point2i cur_pos;

	int im_width;
	int im_height;

	const float min_area = 100 * 100;
	const float max_area = 350 * 350;

	cv::Rect cur_position;
	float cur_rot_degree;
	float cur_scale;
	float delta_rot;
	float delta_scale;
	float mag;

	inline cv::Size scaleSize(cv::Size& sz, float sc) {
		int w = floor(sz.width*sc);
		int h = floor(sz.height*sc);
		return cv::Size(w, h);
	}
	template<class T>
	inline T scaleRect(T& rect, float sc) {
		int x = floor(rect.x*sc);
		int y = floor(rect.y*sc);
		int w = floor(rect.width*sc);
		int h = floor(rect.height*sc);
		return T(x, y, w, h);
	}

	cv::Rect testKCFTracker(const cv::Mat& image, cv::Rect& rect, bool init = false);
	cv::Mat getFeatures(const cv::Mat & patch, cv::Mat& han, int* sizes, bool inithann = false);
	cv::Mat getPixFeatures(const cv::Mat& patch, int* size);
	float subPixelPeak(float left, float center, float right);
	float calcPSR(const cv::Mat& res, cv::Point2i& peak_loc);
protected:
	void estimateLocation(cv::Mat& z, cv::Mat x);
	void estimateScale(cv::Mat& z, cv::Mat& x);

	void updateModel(cv::Mat& image, int polish);	//MATLAB code

	void trainLocation(cv::Mat& x, float train_interp_factor);
	void trainScale(cv::Mat& x, float train_interp_factor);

	void createGaussianPeak(int sizey, int sizex);

	void getTemplates(const cv::Mat& image);

	void getSubWindow(const cv::Mat& image, cv::Size& win0);
	cv::Mat padImage(const cv::Mat& image, int& x1, int& y1, int& x2, int& y2);
	cv::Mat cropImage(const cv::Mat& image, const cv::Point2i& pos, const cv::Size& sz);
	cv::Mat cropImageAffine(const cv::Mat& image, const cv::Point2i& pos, const cv::Size& sz, float scale, float rot);	

	

	cv::Mat hogFeatures;
	cv::Mat _alphaf;
	cv::Mat _y;
	cv::Mat _yf;	//alphaf on f domain
	cv::Mat _z;	//template on time domain
	cv::Mat _zf; //template on f domain
	cv::Mat modelPatch;
	cv::Mat modelPatchf;
	cv::Mat _num;
	cv::Mat _den;
	cv::Mat _labCentroids;
	cv::Rect _roi;

private:
	int size_patch[3];
	int size_scale[3];
	int size_search[3];
	float _scale;
	int _gaussian_size;
	bool _hogfeatures;
	bool _labfeatures;
	bool _rotation;
};