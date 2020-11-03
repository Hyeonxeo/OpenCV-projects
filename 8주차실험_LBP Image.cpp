#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;

void LBP(Mat InputArray, Mat OutputArray) {
	int height = InputArray.rows;
	int width = InputArray.cols;
	int dx[8] = { 0,1,1,1,0,-1,-1,-1 };
	int dy[8] = { -1,-1,0,1,1,1,0,-1 };
	
	int i, j, k;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			int lbp[8] = { 0 };
			//lbp array of specific pixel
			for (k = 0; k < 8; k++) {
				int y = i + dy[k];
				int x = j + dx[k];
				if (y >= 0 && y < height && x >= 0 && x < width) {
					if (InputArray.at<uchar>(i, j) < InputArray.at<uchar>(y, x)) {
						lbp[k] = 1;
					}
					else {
						lbp[k] = 0;
					}
				}
			}
			//acquring lbp property
			int lbp_property = 0;
			int factor = 1;
			for (k = 7; k >=0; k--) {
				lbp_property += (lbp[k] * factor);
				factor *= 2;
			}
			OutputArray.at<uchar>(i, j) = (uchar)lbp_property;
		}
	}

}

void main() {
	Mat img = imread("testimage.jpg", IMREAD_GRAYSCALE);
	Mat img_LBP(img.rows, img.cols, CV_8UC1);

	LBP(img, img_LBP);
	imshow("result", img_LBP);
	waitKey(0);


}