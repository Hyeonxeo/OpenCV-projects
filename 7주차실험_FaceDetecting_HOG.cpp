#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <math.h>
#define PI 3.141592

using namespace std;
using namespace cv;

int R_height, R_width;
int T_height, T_width;
int template_width, template_height;

typedef struct {
	double magnitude, phase;

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
			G[y][x].magnitude = sqrt(convX * convX + convY * convY);
			G[y][x].phase = atan2(convY, convX) * 180 / PI;
			if (G[y][x].phase < 0) {
				G[y][x].phase += 180;
			}
		}
	}
}


void GetHOG(gradient** G, double*** HOG,int start_y,int start_x) { //window 8x8
	int i, j,y,x;

	for (y = start_y; y < start_y+R_height-4; y+=4) {
		for (x = start_x; x < start_x + R_width-4; x += 4) {
			if (y >= 0 && y < T_height && x >= 0 && x < T_width) {
				double min = 1000000.0, max = 0.0;
				int Y = (y - start_y) / 4;
				int X = (x - start_x) / 4;
				for (i = y; i < y + 8; i++) {
					for (j = x; j < x + 8; j++) {
						if (i < T_height && j < T_width) {
							if ((G[i][j].phase >= 170 && G[i][j].phase <= 180) || (G[i][j].phase >= 0 && G[i][j].phase < 10)) {//0
								HOG[Y][X][0] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 10 && G[i][j].phase < 30) {//20
								HOG[Y][X][1] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 30 && G[i][j].phase < 50) {//40
								HOG[Y][X][2] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 50 && G[i][j].phase < 70) {//60
								HOG[Y][X][3] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 70 && G[i][j].phase < 90) {//80
								HOG[Y][X][4] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 90 && G[i][j].phase < 110) {//100
								HOG[Y][X][5] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 110 && G[i][j].phase < 130) {//120
								HOG[Y][X][6] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 130 && G[i][j].phase < 150) {//140
								HOG[Y][X][7] += G[i][j].magnitude;
							}
							else if (G[i][j].phase >= 150 && G[i][j].phase < 170) {//160
								HOG[Y][X][8] += G[i][j].magnitude;
							}
						}
					}
				}

				for (i = 0; i < 9; i++) {
					if (min > HOG[Y][X][i]) {
						min = HOG[Y][X][i];
					}
					if (max < HOG[Y][X][i]) {
						max = HOG[Y][X][i];
					}
				}
				double delta = max - min;
				for (i = 0; i < 9; i++) {
					HOG[Y][X][i] = 255 * (HOG[Y][X][i] - min) / delta;
				}

			}
		}
	}
	

}

void GetSimilarity(gradient** G, double** S,double***Ref_HOG,int start_y,int start_x) {
	int i, j, k;
	double*** Tar_HOG = (double***)calloc(template_height, sizeof(double**));
	for (i = 0; i < template_height; i++) {
		Tar_HOG[i] = (double**)calloc(template_width, sizeof(double*));
		for (j = 0; j < template_width; j++) {
			Tar_HOG[i][j] = (double*)calloc(9, sizeof(double));
		}
	}
	GetHOG(G, Tar_HOG, start_y, start_x);
	
	double Relation = 0.0;
	for (i = 0; i < template_height; i++) {
		for (j = 0; j < template_width; j++) {
			for (k = 0; k < 9; k++) {
				Relation += fabs(Ref_HOG[i][j][k] - Tar_HOG[i][j][k]);		//다른방법으로도 distance 구해보기 -> eucladian method는 잡음에 약함
			}
		}
	}
	Relation =Relation/ (double)(template_height * template_width * 9);
	S[start_y][start_x] = Relation;

	//free memory
	for (i = 0; i < template_height; i++) {
		for (j = 0; j < template_width; j++) {
			free(Tar_HOG[i][j]);
		}
		free(Tar_HOG[i]);
	}
	free(Tar_HOG);

}

void Detecting(Mat Reference, Mat Target) {
	R_height = Reference.rows;
	R_width = Reference.cols;
	T_height = Target.rows;
	T_width = Target.cols;

	int i, j;
	gradient** R_Gradient = (gradient**)malloc(sizeof(gradient*) * R_height);
	for (i = 0; i < R_height; i++) {
		R_Gradient[i] = (gradient*)malloc(sizeof(gradient) * R_width);
	}
	gradient** T_Gradient = (gradient**)malloc(sizeof(gradient*) * T_height);
	for (i = 0; i < T_height; i++) {
		T_Gradient[i] = (gradient*)malloc(sizeof(gradient) * T_width);
	}
	GetGradient(Reference, R_Gradient);
	GetGradient(Target, T_Gradient);

	//template HOG
	template_width = (R_width / 4) - 1;			//HOG window: width=8 & height=8, 
	template_height=(R_height / 4) - 1;
	
	double*** Ref_HOG = (double***)calloc(template_height,sizeof(double**));
	for (i = 0; i < template_height; i++) {
		Ref_HOG[i] = (double**)calloc(template_width, sizeof(double*));
		for (j = 0; j < template_width; j++) {
			Ref_HOG[i][j] = (double*)calloc(9, sizeof(double));
		}
	}

	GetHOG(R_Gradient, Ref_HOG,0,0);
	
	//compare with target
	double** Similarity = (double**)malloc(T_height * sizeof(double*));		// relativity array 
	for (i = 0; i < T_height; i++) {
		Similarity[i] = (double*)malloc(T_width * sizeof(double));
	}


	double min = 100000.0, max = 0.0;
	for (i = 0; i < T_height; i++) {			//Search all pixels of Target 
		for (j = 0; j < T_width; j++) {
			GetSimilarity(T_Gradient, Similarity, Ref_HOG,i,j);
			if (max < Similarity[i][j]) {
				max = Similarity[i][j];
			}
			if (min > Similarity[i][j]) {
				min = Similarity[i][j];
			}
		}
	}
	double delta = max - min;
	for (i = 0; i < T_height; i++) {
		for (j = 0; j < T_width; j++) {
			Similarity[i][j] = 255 *(Similarity[i][j] - min) / delta;
		}
	}
	
	Mat result_S(T_height, T_width, CV_8UC1);

	for (i = 0; i < T_height; i++) {
		for (j = 0; j < T_width; j++) {
			result_S.at<uchar>(i, j) = (uchar)Similarity[i][j];
			if (Similarity[i][j] < 30) {
				rectangle(Target, Point(j, i), Point(j + R_width, i + R_height), Scalar(255, 0, 0), 2, 8, 0);
			}
		}
	}
	imshow("resulta", result_S);
	imshow("reuslt", Target);
	waitKey(0);






	//free memory
	for (i = 0; i < T_height; i++) {
		free(Similarity[i]);
	}
	free(Similarity);
	////////////
	for(i = 0; i < template_height; i++) {
		for (j = 0; j < template_width; j++) {
			free(Ref_HOG[i][j]);
		}
		free(Ref_HOG[i]);
	}
	free(Ref_HOG);
	//////////
	for (i = 0; i < R_height; i++) {
		free(R_Gradient[i]);
	}
	free(R_Gradient);
	for (i = 0; i < T_height; i++) {
		free(T_Gradient[i]);
	}
	free(T_Gradient);
}


void main() {
	Mat img_ref = imread("face_ref.bmp", IMREAD_GRAYSCALE);
	Mat img_tar = imread("face_tar.bmp", IMREAD_GRAYSCALE);

	
	Detecting(img_ref, img_tar);

}