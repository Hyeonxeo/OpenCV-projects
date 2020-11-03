#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdlib.h>
#define PI 3.141592

using namespace cv;


void FindHOG(Mat ImgSrc, double* HOGmatrix) {
	int height = ImgSrc.rows;			//128
	int width = ImgSrc.cols;			//64

	int FilterX[3][3] = { -1,0,1,-1,0,1,-1,0,1 };
	int FilterY[3][3] = { -1,-1,-1,0,0,0,1,1,1 };
	double* magnitude = (double*)malloc(sizeof(double) * height * width);
	double* phase = (double*)malloc(sizeof(double) * height * width);
	int y, x, i, j, convX, convY;
	double sum = 0.0, radian = 0.0, degree = 0.0;


	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			convX = 0;
			convY = 0;
			for (i = y - 1; i <= y + 1; i++) {
				for (j = x - 1; j <= x + 1; j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						convX += ImgSrc.at<uchar>(i, j) * FilterX[i - y + 1][j - x + 1];
						convY += ImgSrc.at<uchar>(i, j) * FilterY[i - y + 1][j - x + 1];
					}
				}
			}
			sum = sqrt(convX * convX + convY * convY);
			magnitude[width * y + x] = sum;
			radian = atan2(convY, convX);
			degree = radian * 180 / PI;
			phase[width * y + x] = degree;
		}
	}

	for (y = 0; y < 15; y++) {
		for (x = 0; x < 7; x++) {
			double min = 1000000.0, max = 0.0;

			for (i = 8 * y; i < 8 * y + 16; i++) {
				for (j = 8 * x; j < 8* x + 16; j++) {
					if ((phase[i * width + j] >= 170 && phase[i * width + j] <= 180) || (phase[i * width + j] >= 0 && phase[i * width + j] < 10)) {//0
						HOGmatrix[9 * (7 * y + x) + 0] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 10 && phase[i * width + j] < 30) {//20
						HOGmatrix[9 * (7 * y + x) + 1] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 30 && phase[i * width + j] < 50) {//40
						HOGmatrix[9 * (7 * y + x) + 2] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 50 && phase[i * width + j] < 70) {//60
						HOGmatrix[9 * (7 * y + x) + 3] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 70 && phase[i * width + j] < 90) {//80
						HOGmatrix[9 * (7 * y + x) + 4] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 90 && phase[i * width + j] < 110) {//100
						HOGmatrix[9 * (7 * y + x) + 5] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 110 && phase[i * width + j] < 130) {//120
						HOGmatrix[9 * (7 * y + x) + 6] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 130 && phase[i * width + j] < 150) {//140
						HOGmatrix[9 * (7 * y + x) + 7] += magnitude[i * width + j];
					}
					else if (phase[i * width + j] >= 150 && phase[i * width + j] < 170) {//160
						HOGmatrix[9 * (7 * y + x) + 8] += magnitude[i * width + j];
					}
				}
			}

			for (i = 0; i < 9; i++) {
				if (min > HOGmatrix[9 * (7 * y + x) + i]) {
					min = HOGmatrix[9 * (7 * y + x) + i];
				}
				if (max < HOGmatrix[9 * (7 * y + x) + i]) {
					max = HOGmatrix[9 * (7 * y + x) + i];
				}
			}
			for (i = 0; i < 9; i++) {
				HOGmatrix[9 * (7 * y + x) + i] = 255 * (HOGmatrix[9 * (7 * y + x) + i] - min) / (max - min);
			}

		}
	}

	free(magnitude);
	free(phase);
}


int main() {
	Mat Img_Assignment = imread("assignment3.bmp", IMREAD_GRAYSCALE);
	double* HOG_Assignment = (double*)calloc(945, sizeof(double));

	FindHOG(Img_Assignment, HOG_Assignment);
	
	Mat Img_Compare1 = imread("compare1.bmp", IMREAD_GRAYSCALE);
	Mat Img_Compare2 = imread("compare2.bmp", IMREAD_GRAYSCALE);
	double* HOG_Compare1 = (double*)calloc(945, sizeof(double));
	double* HOG_Compare2 = (double*)calloc(945, sizeof(double));

	FindHOG(Img_Compare1, HOG_Compare1);
	FindHOG(Img_Compare2, HOG_Compare2);

	double Relation1=0.0, Relation2=0.0;

	for (int i = 0; i < 15; i++) {
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 9; k++) {
				Relation1 += fabs(HOG_Assignment[9 * (i * 7 + j) + k] - HOG_Compare1[9 * (i * 7 + j) + k]);
				Relation2+= fabs(HOG_Assignment[9 * (i * 7 + j) + k] - HOG_Compare2[9 * (i * 7 + j) + k]);
			}
		}
	}
	Relation1 /= 945;
	Relation2 /= 945;
	printf("Relation1 = %lf\nRelation2 = %lf", Relation1, Relation2);
	
	free(HOG_Assignment);
	free(HOG_Compare1);
	free(HOG_Compare2);
}