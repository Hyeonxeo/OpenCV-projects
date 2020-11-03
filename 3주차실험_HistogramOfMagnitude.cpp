#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.141592

using namespace cv;


int main() {
	Mat ImgGray = imread("test.jpg", IMREAD_GRAYSCALE);
	int height, width;
	height = ImgGray.rows;
	width = ImgGray.cols;


	Mat ImgEdge(height, width, CV_8UC1);
	int FilterX[3][3] = { -1,0,1,-1,0,1,-1,0,1 };
	int FilterY[3][3] = { -1,-1,-1,0,0,0,1,1,1 };
	double histogram[9] = { 0.0 };		//0~10=0/10~30=20/30~50=40/50~70=60/70~90=80/90~110=100/110~130=120/130~150=140/150~170=160/170~180=0
	double* magnitude = (double*)malloc(sizeof(double) * height * width);

	int y, x, i, j, convX, convY;
	double sum, min = 10000.0, max = 0.0, radian = 0.0, degree = 0.0;;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			convX = 0;
			convY = 0;
			for (i = y - 1; i <= y + 1; i++) {
				for (j = x - 1; j <= x + 1; j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						convX += ImgGray.at<uchar>(i, j) * FilterX[i - y + 1][j - x + 1];
						convY += ImgGray.at<uchar>(i, j) * FilterY[i - y + 1][j - x + 1];
					}
				}
			}
			radian = atan2(convY, convX);
			degree = radian * 180 / PI;
			if (degree < 0) {
				degree += 180;
			}
			sum = sqrt(convX * convX + convY * convY);

			if ((degree >= 170 && degree <= 180) || (degree >= 0 && degree < 10)) {//0
				histogram[0] += sum;
			}
			else if (degree >= 10 && degree < 30) {//20
				histogram[1] += sum;
			}
			else if (degree >= 30 && degree < 50) {//40
				histogram[2] += sum;
			}
			else if (degree >= 50 && degree < 70) {//60
				histogram[3] += sum;
			}
			else if (degree >= 70 && degree < 90) {//80
				histogram[4] += sum;
			}
			else if (degree >= 90 && degree < 110) {//100
				histogram[5] += sum;
			}
			else if (degree >= 110 && degree < 130) {//120
				histogram[6] += sum;
			}
			else if (degree >= 130 && degree < 150) {//140
				histogram[7] += sum;
			}
			else if (degree >= 150 && degree < 170) {//160
				histogram[8] += sum;
			}
		}
	}

	FILE* fp = fopen("histogram of phase and magnitude.txt", "w");
	
	for (i = 0; i < 9; i++) {
		fprintf(fp, "%d -- %lf\n", 20 * i, histogram[i]);
		printf("%d -- %lf\n", 20 * i, histogram[i]);
	}
	fclose(fp);

	imshow("gray", ImgGray);

	waitKey(0);
}