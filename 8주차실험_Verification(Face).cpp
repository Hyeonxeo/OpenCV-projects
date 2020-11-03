#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#define PI 3.141592
using namespace cv;
using namespace cv::ml;
using namespace std;

int method_flag = 0;
int Ref_Histogram[7][7][256] = { 0 };
int Tar_Histogram[7][7][256] = { 0 };


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
			if (method_flag == 1) {
				G[y][x].magnitude = sqrt(convX * convX + convY * convY);
			}
			else if (method_flag == 2) {
				G[y][x].phase = atan2(convY, convX) * 180 / PI;
				if (G[y][x].phase < 0) {
					G[y][x].phase += 180;
				}
			}
		}
	}
}

void LBP(Mat InputArray, Mat OutputArray) {
	int height = InputArray.rows;
	int width = InputArray.cols;
	int dx[8] = { 0,1,1,1,0,-1,-1,-1 };
	int dy[8] = { -1,-1,0,1,1,1,0,-1 };
	int lbp[8] = { 0 };
	int i, j, k;
	if(method_flag==1 || method_flag==2){
		gradient** G = (gradient**)malloc(height * sizeof(gradient*));
		for (i = 0; i < height; i++) {
			G[i] = (gradient*)malloc(width * sizeof(gradient));
		}
		GetGradient(InputArray, G);


		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				int lbp[8] = { 0 };

				//lbp array of specific pixel
				for (k = 0; k < 8; k++) {
					int y = i + dy[k];
					int x = j + dx[k];
					if (y >= 0 && y < height && x >= 0 && x < width) {
						if (method_flag == 1) {
							if (G[i][j].magnitude < G[y][x].magnitude) {
								lbp[k] = 1;
							}
							else {
								lbp[k] = 0;
							}
						}
						else if (method_flag == 2) {
							float factor = fabs(cos(G[i][j].phase - G[y][x].phase));
							if (factor >= 0.9) {
								lbp[k] = 1;
							}
							else if (factor < 0.9) {
								lbp[k] = 0;
							}
						}
					}
				}
				//acquring lbp property
				int lbp_property = 0;
				int factor = 1;
				for (k = 7; k >= 0; k--) {
					lbp_property += (lbp[k] * factor);
					factor *= 2;
				}
				OutputArray.at<uchar>(i, j) = (uchar)lbp_property;
			}
		}
		for (i = 0; i < height; i++) {
			free(G[i]);
		}
		free(G);


	}
	else {
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
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
				for (k = 7; k >= 0; k--) {
					lbp_property += (lbp[k] * factor);
					factor *= 2;
				}
				OutputArray.at<uchar>(i, j) = (uchar)lbp_property;
			}
		}
	}
}

void GetHist(Mat InputArray, int histogram[][7][256]) {
	int Matcher_height, Matcher_width,Window_height, Window_width;
	Matcher_height = InputArray.rows;			//size of face pic 
	Matcher_width = InputArray.cols;
	Window_height = Matcher_height / 4;			//7개의 block
	Window_width = Matcher_width / 4;


	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			double min = 100000.0, max = 0.0;
			int start_y = i * (Window_height / 2);
			int start_x = j * (Window_width / 2);
			for (int y = start_y; y < start_y + Window_height; y++) {
				for (int x = start_x; x < start_x + Window_width; x++) {
					int k = (int)InputArray.at<uchar>(y, x);
					histogram[i][j][k] += 1;
				}
			}

			for (int k = 0; k < 256; k++) {
				if (max < histogram[i][j][k]) {
					max = histogram[i][j][k];
				}
				if (min > histogram[i][j][k]) {
					min = histogram[i][j][k];
				}
			}
			int delta = max - min;
			for (int k = 0; k < 256; k++) {
				histogram[i][j][k] = 255 * (histogram[i][j][k] - min) / delta;	//각 block을 255으로 normalization
			}

		}
	}

}

double GetRelation(int histogram1[][7][256], int histogram2[][7][256]) {
	int factor =  7 *7* 256;
	double Relation = 0.0;
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 256; k++) {
				int bunmo = (histogram1[i][j][k] + histogram2[i][j][k]);
				if (bunmo != 0) {
					Relation += ((histogram1[i][j][k] - histogram2[i][j][k]) * (histogram1[i][j][k] - histogram2[i][j][k]) / bunmo);// chi_squared distance
				}
				else {
					Relation += 0;
				}
			}
		}
	}
	Relation /= factor;
	return Relation;

}


void main() {
	Mat Face_source=imread("facematch2.jpg",IMREAD_COLOR);
	Mat Ref_Face;
	Mat Face_source_Target = imread("facematch1.jpg", IMREAD_COLOR);
	Mat Tar_Face;
	
	CascadeClassifier cascade;
	cascade.load("C:\\Users\\illus\\Downloads\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml");
	
	vector<Rect>face;
	cascade.detectMultiScale(Face_source, face, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));
	vector<Rect>face_tar;
	cascade.detectMultiScale(Face_source_Target, face_tar, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

	printf("Select Method\n1 : Magnitude lbp\n2 : cosine similarity\n");
	scanf_s("%d", &method_flag);


	if (face.size() == 1) {
		Point lb(face[0].x + face[0].width, face[0].y + face[0].height);
		Point tr(face[0].x, face[0].y);
		Ref_Face = Face_source(Range(face[0].y, face[0].y + face[0].height), Range(face[0].x, face[0].x + face[0].width));
		rectangle(Face_source, lb, tr, Scalar(255, 0, 0), 2, 8, 0);

		Mat Ref_Face_gray;
		Mat Ref_LBP(Ref_Face.rows, Ref_Face.cols, CV_8UC1);
		cvtColor(Ref_Face, Ref_Face_gray, COLOR_BGR2GRAY);
		LBP(Ref_Face_gray, Ref_LBP);
		GetHist(Ref_LBP, Ref_Histogram);
		imshow("lbp_face", Ref_LBP);
		imshow("dadfasd", Ref_Face);
		waitKey(0);
	}
	else {
		printf("no face_0");
	}

	double relation_Ref2Tar=0.0;
	
	if (face_tar.size() == 1) {
		Point lb(face_tar[0].x + face_tar[0].width, face_tar[0].y + face_tar[0].height);
		Point tr(face_tar[0].x, face_tar[0].y);
		Tar_Face = Face_source_Target(Range(face_tar[0].y, face_tar[0].y + face_tar[0].height), Range(face_tar[0].x, face_tar[0].x + face_tar[0].width));

		Mat Tar_Face_gray;
		Mat Tar_LBP(Tar_Face.rows, Tar_Face.cols, CV_8UC1);
		cvtColor(Tar_Face, Tar_Face_gray, COLOR_BGR2GRAY);
		LBP(Tar_Face_gray, Tar_LBP);
		GetHist(Tar_LBP, Tar_Histogram);
		imshow("lbp_face", Tar_LBP);
		imshow("dadfasd", Tar_Face);
		waitKey(0);

		relation_Ref2Tar = GetRelation(Ref_Histogram, Tar_Histogram);
	
	}
	else if(face_tar.size() == 0) {
		printf("no face");
	}


	printf("%lf", relation_Ref2Tar);

}

	