#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace cv;


int main() {
	Mat ImgGray = imread("test.jpg", IMREAD_GRAYSCALE);
	int height, width;
	height = ImgGray.rows;
	width = ImgGray.cols;


	Mat ImgEdge(height, width, CV_8UC1);
	int FilterX[3][3] = { -1,0,1,-1,0,1,-1,0,1 };
	int FilterY[3][3] = { -1,-1,-1,0,0,0,1,1,1 };	
	double* magnitude = (double*)malloc(sizeof(double) * height * width);

	int y, x,i,j,convX,convY;
	double sum,min=10000.0,max=0.0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			convX = 0;
			convY = 0;
			for (i = y - 1; i<=y + 1; i++) {
				for (j = x - 1;j<= x + 1; j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						convX += ImgGray.at<uchar>(i, j) * FilterX[i - y + 1][j - x + 1];
						convY += ImgGray.at<uchar>(i, j) * FilterY[i - y + 1][j - x + 1];
					}
				}
			}
			sum = sqrt(convX * convX + convY * convY);
			magnitude[width * y + x] = sum;
			if (min > sum) {
				min = sum;
			}
			if (max < sum) {
				max = sum;
			}
		}
	}

	//normalization
	double delta = max - min;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			ImgEdge.at<uchar>(y, x) = 255 * (magnitude[width * y + x] - min) / delta;
		}
	}
	free(magnitude);
	imshow("gray", ImgGray);
	imshow("edge", ImgEdge);
	waitKey(0);
}