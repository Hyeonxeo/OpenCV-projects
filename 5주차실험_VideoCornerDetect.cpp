#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>
#define PI 3.141592

using namespace cv;


void LowPassFilter(Mat InputArray, Mat OutputArray) {
	int i, j, x, y;
	double GaussianFilter[5][5] = { {0.0029,0.0133,0.0219,0.0133,0.0029},
								  {0.0133,0.0596,0.0983,0.0596,0.0133},
								  {0.0219,0.0983,0.1621,0.0983,0.0219},
								  {0.0133,0.0596,0.0983,0.0596,0.0133},
								  {0.0029,0.0133,0.0219,0.0133,0.0029} };

	int height = InputArray.rows;
	int width = InputArray.cols;
	double FilteredPixel;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			FilteredPixel = 0.0;
			for (i = y - 2; i < y + 2; i++) {
				for (j = x - 2; j < x + 2; j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						FilteredPixel += InputArray.at<uchar>(i, j) * GaussianFilter[i - y + 2][j - x + 2];
					}
				}
			}
			OutputArray.at<uchar>(y, x) = (uchar)FilteredPixel;
		}
	}
}

typedef struct {
	long long Ix, Iy;
	double R;
}gradient;

void GetGradient(Mat InputArray, gradient** G) {
	int y, x, i, j, convX, convY;
	int height = InputArray.rows;
	int width = InputArray.cols;
	int FilterX[3][3] = { -1,0,1,-1,0,1,-1,0,1 };
	int FilterY[3][3] = { -1,-1,-1,0,0,0,1,1,1 };

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			convX = 0;
			convY = 0;
			for (i = y - 1; i <= y + 1; i++) {
				for (j = x - 1; j <= x + 1; j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						convX += InputArray.at<uchar>(i, j) * FilterX[i - y + 1][j - x + 1];
						convY += InputArray.at<uchar>(i, j) * FilterY[i - y + 1][j - x + 1];
					}
				}
			}
			G[y][x].Ix = convX;
			G[y][x].Iy = convY;
		}
	}
}

void HarrisStephenCornerDetecting(Mat InputArray,Mat OutputArray) {
	int x, y, i, j;
	int height = InputArray.rows;
	int width = InputArray.cols;
	gradient** Gradient = (gradient**)malloc(sizeof(gradient) * height);
	for (i = 0; i < height; i++) {
		Gradient[i] = (gradient*)malloc(sizeof(gradient) * width);
	}
	GetGradient(InputArray, Gradient);

	int SizeOfWindow=5;
	
	long long M1, M2, M3, M4;
	double min = 0.0, max = 0.0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			M1 = M2 = M3 = M4 = 0.0;
			for (i = y - (SizeOfWindow / 2); i <= y + (SizeOfWindow / 2); i++) {
				for (j = x - (SizeOfWindow / 2); j <= x + (SizeOfWindow / 2); j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						M1 += Gradient[i][j].Ix * Gradient[i][j].Ix;
						M2 += Gradient[i][j].Ix * Gradient[i][j].Iy;
						M3 = M2;
						M4 += Gradient[i][j].Iy * Gradient[i][j].Iy;
					}
				}
			}
			Gradient[y][x].R = ((M1 * M4) - (M2 * M3)) - (0.05 * (M1 + M4) * (M1 + M4));
			if (min > Gradient[y][x].R) {
				min = Gradient[y][x].R;
			}
			if (max < Gradient[y][x].R) {
				max = Gradient[y][x].R;
			}
			//printf("%d\n", Gradient[y][x].R);
		}

	}
	double delta = max - min;

	//normalize
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			Gradient[y][x].R = 100000 * (Gradient[y][x].R - min) / delta;
		}
	}


	Scalar c;
	Point pCenter;
	int radius = 1;
	int threshold =39500;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (Gradient[y][x].R > threshold) {
				pCenter.x = x;
				pCenter.y = y;
				c.val[0] = 0;
				c.val[1] = 0;
				c.val[2] = 255;
				circle(OutputArray, pCenter, radius, c, 2, 8, 0);
			}
		}
	}


	for (i = 0; i < height; i++) {
		free(Gradient[i]);
	}
	free(Gradient);
}


void main() {
	VideoCapture capture(0);
	Mat frame;
	Mat frameGray;

	while (true) {
		capture >> frame;
		Mat frameFilter(frame.rows, frame.cols, CV_8UC1);
		Mat cornerMap(frame.rows, frame.cols, CV_8UC3);
		cornerMap = frame.clone();
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);
		LowPassFilter(frameGray, frameFilter);
		HarrisStephenCornerDetecting(frameFilter, cornerMap);
	
		imshow("Corner Detection", cornerMap);
		if (waitKey(30) >= 0)break;
		
	}

}