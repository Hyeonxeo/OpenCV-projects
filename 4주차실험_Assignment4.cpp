#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#define PI 3.141592

using namespace cv;
using namespace std;


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
	double R,magnitude,phase;
	
}gradient;
typedef struct {
	int x, y;
	double HOG[9] = { 0 };
}corner;


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
			G[y][x].magnitude = sqrt(convX * convX + convY * convY);
			G[y][x].phase = atan2(convY, convX) * 180 / PI;
			if (G[y][x].phase < 0) {
				G[y][x].phase += 180;
			}
		}
	}
}

void HarrisStephenCornerDetecting(Mat InputArray,gradient**Gradient) {
	int x, y, i, j;
	int height = InputArray.rows;
	int width = InputArray.cols;
	
	GetGradient(InputArray, Gradient);

	int SizeOfWindow=5;
	//"Size of Window :  5 "
	
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

}


void getHOG(vector<corner>&C, gradient** G, int count, int height, int width) {
	int Y = C[count].y;
	int X = C[count].x;

	int i, j;
	double min = 10000000.0, max = 0.0;
	for (i = Y - 8; i < Y + 8; i++) {					//corner라고 인지한 점을 중심으로 16x16 hog
		for (j = X - 8; j < X + 8; j++) {
			if (i >= 0 && i < height && j >= 0 && j < width) {
				if ((G[i][j].phase >= 170 && G[i][j].phase <= 180) || (G[i][j].phase >= 0 && G[i][j].phase < 10)) {//0
					C[count].HOG[0] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 10 && G[i][j].phase < 30) {//20
					C[count].HOG[1] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 30 && G[i][j].phase < 50) {//40
					C[count].HOG[2] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 50 && G[i][j].phase < 70) {//60
					C[count].HOG[3] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 70 && G[i][j].phase < 90) {//80
					C[count].HOG[4] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 90 && G[i][j].phase < 110) {//100
					C[count].HOG[5] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 110 && G[i][j].phase < 130) {//120
					C[count].HOG[6] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 130 && G[i][j].phase < 150) {//140
					C[count].HOG[7] += G[i][j].magnitude;
				}
				else if (G[i][j].phase >= 150 && G[i][j].phase < 170) {//160
					C[count].HOG[8] += G[i][j].magnitude;
				}
			}
		}
	}

	for (i = 0; i < 9; i++) {
		if (min > C[count].HOG[i]) {
			min = C[count].HOG[i];
		}
		if (max < C[count].HOG[i]) {
			max = C[count].HOG[i];
		}
	}
	double delta = max - min;
	for (i = 0; i < 9; i++) {
		C[count].HOG[i] = 255 * (C[count].HOG[i] - min) / delta;
	}
	

}


void CompareCorner(Mat InputArray1, Mat InputArray2) {
	int i, j,y,x;
	int height_1 = InputArray1.rows;
	int width_1 = InputArray1.cols;
	gradient** Gradient1 = (gradient**)malloc(sizeof(gradient) * height_1);
	for (i = 0; i < height_1; i++) {
		Gradient1[i] = (gradient*)malloc(sizeof(gradient) * width_1);
	}

	int height_2 = InputArray2.rows;
	int width_2 = InputArray2.cols;
	gradient** Gradient2 = (gradient**)malloc(sizeof(gradient) * height_2);
	for (i = 0; i < height_2; i++) {
		Gradient2[i] = (gradient*)malloc(sizeof(gradient) * width_2);
	}

	HarrisStephenCornerDetecting(InputArray1, Gradient1);
	HarrisStephenCornerDetecting(InputArray2, Gradient2);



	Mat Img_result_1 = imread("ref.bmp", IMREAD_COLOR);
	Mat Img_result_2 = imread("tar.bmp", IMREAD_COLOR);
	
	Scalar c;
	Point pCenter;
	int radius = 3;
	int threshold=39000,count1=0,count2=0;

	for (y = 0; y < height_1; y++) {
		for (x = 0; x < width_1; x++) {
			if (Gradient1[y][x].R > threshold) {
				pCenter.x = x;
				pCenter.y = y;
				c.val[0] = 0;
				c.val[1] = 0;
				c.val[2] = 255;
				circle(Img_result_1, pCenter, radius, c, 2, 8, 0);
				count1++;
			}
		}
	}
	for (y = 0; y < height_2; y++) {
		for (x = 0; x < width_2; x++) {
			if (Gradient2[y][x].R > threshold) {
				pCenter.x = x;
				pCenter.y = y;
				c.val[0] = 0;
				c.val[1] = 0;
				c.val[2] = 255;
				circle(Img_result_2, pCenter, radius, c, 2, 8, 0);
				count2++;
			}
		}
	}

	vector<corner> Corner1(count1);
	vector<corner>Corner2(count2);
	count1 = count2 = 0;
	for (y = 0; y < height_1; y++) {
		for (x = 0; x < width_1; x++) {
			if (Gradient1[y][x].R > threshold) {
				Corner1[count1].x = x;
				Corner1[count1].y = y;
				getHOG(Corner1, Gradient1, count1, height_1, width_1);
				count1++;
			}
		}
	}
	for (y = 0; y < height_2; y++) {
		for (x = 0; x < width_2; x++) {
			if (Gradient2[y][x].R > threshold) {
				Corner2[count2].x = x;
				Corner2[count2].y = y;
				getHOG(Corner2, Gradient2, count2, height_2, width_2);
				count2++;
			}
		}
	}
	
	

	int Height = height_1;
	int Width = width_1 + width_2;
	Mat Img_Result(Height,Width , CV_8UC3);
	for (y = 0; y < Height; y++) {
		for (x = 0; x < width_1; x++) {
			for (i = 0; i < 3; i++) {
				Img_Result.at<Vec3b>(y, x)[i] = Img_result_1.at<Vec3b>(y, x)[i];
			}
		}
	}
	for (y = 0; y < Height; y++) {
		for (x = width_1; x < Width; x++) {
			for (i = 0; i < 3; i++) {
				Img_Result.at<Vec3b>(y, x)[i] = Img_result_2.at<Vec3b>(y, (x-width_1))[i];
			}
		}
	}

	for (i = 0; i < count1; i++) {
		for (j = 0; j < count2; j++) {
			double Relation = 0.0;
			for (x = 0; x < 9; x++) {
				Relation += fabs(Corner1[i].HOG[x] - Corner2[j].HOG[x]);
				
			}
			Relation /= 9;
			//printf("%lf \n", Relation);
			if (Relation < 10) {
				int x1 = Corner1[i].x, y1 = Corner1[i].y;
				int x2 = Corner2[j].x + width_1, y2 = Corner2[j].y;
				line(Img_Result, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2, 9, 0);
			}

		}
	}

	imshow("result", Img_Result);
	waitKey(0);




	for (i = 0; i < height_2; i++) {
		free(Gradient2[i]);
	}
	free(Gradient2);

	for (i = 0; i < height_1; i++) {
		free(Gradient1[i]);
	}
	free(Gradient1);

}


int main() {
	Mat Img_reference = imread("ref.bmp", IMREAD_GRAYSCALE);
	int height_R = Img_reference.rows;
	int width_R = Img_reference.cols;
	Mat Img_ref(height_R, width_R, CV_8UC1);
	Mat Img_target = imread("tar.bmp", IMREAD_GRAYSCALE);
	int height_T = Img_target.rows;
	int width_T = Img_target.cols;
	Mat Img_tar(height_T, width_T, CV_8UC1);

	LowPassFilter(Img_reference,Img_ref);
	LowPassFilter(Img_target,Img_tar);


	CompareCorner(Img_ref, Img_tar);




}