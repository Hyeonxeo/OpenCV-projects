#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#define PI 3.141592
using namespace cv;


int main() {
	Mat img_src = imread("test3.jpg", IMREAD_COLOR);
	double degree;
	printf("degree : ");
	scanf("%f", &degree);

	double radian = (-1)*degree * PI / 180;

	int height = img_src.rows, width = img_src.cols;
	int cntX = width / 2, cntY = height / 2;

	Mat img_result(height, width, CV_8UC3);
	
	int y, x, i;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			int dy = y - cntY, dx = x - cntX;
			int dY = (dx * sin(radian)) + (dy * cos(radian));
			int dX = (dx * cos(radian)) - (dy * sin(radian));
			int Y = cntY + dY, X = cntX + dX;

			//printf("%d,%d - %d,%d\n", y, x, Y, X);

			if (X >= 0 && X < width && Y >= 0 && Y < height) {
				for (i = 0; i < 3; i++) {
					img_result.at<Vec3b>(y, x)[i] = img_src.at<Vec3b>(Y, X)[i];
				}
			}
			else {
				for (i = 0; i < 3; i++) {
					img_result.at<Vec3b>(y, x)[i] = 25;
				}
			}

		}
	}

	imshow("original", img_src);
	imshow("result", img_result);

	waitKey(0);
	return 0;
}