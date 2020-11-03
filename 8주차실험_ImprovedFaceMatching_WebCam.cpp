//256-dim histogram -->> 59-dim histogram
//1 : lbp of magnitude   
//OR   
//2 : cosine similarity 

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#define PI 3.141592
using namespace cv;
using namespace cv::ml;
using namespace std;

int Ref_Histogram[7][7][59] = { 0 };
int method_flag = 0;
const char lookup[256]{
	0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,
	11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,15,
	16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	17,58,58,58,58,58,58,58,18,58,58,58,19,58,20,21,
	22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,
	29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,33,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,
	36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,
	42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,
	47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57
};

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
	int i, j, k;

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
						if (G[i][j].magnitude<G[y][x].magnitude) {
							lbp[k] = 1;
						}
						else {
							lbp[k] = 0;
						}
					}
					else if (method_flag == 2) {
						float factor = fabs(cos(G[i][j].phase - G[y][x].phase));
						if (factor >= 0.941) {
							lbp[k] = 1;
						}
						else if (factor < 0.941) {
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


void GetHist(Mat InputArray, int histogram[][7][59]) {
	int Matcher_height, Matcher_width, Window_height, Window_width;
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
					int k = lookup[InputArray.at<uchar>(y, x)];
					histogram[i][j][k] += 1;
				}
			}

			for (int k = 0; k < 59; k++) {
				if (max < histogram[i][j][k]) {
					max = histogram[i][j][k];
				}
				if (min > histogram[i][j][k]) {
					min = histogram[i][j][k];
				}
			}
			int delta = max - min;
			for (int k = 0; k < 59; k++) {
				histogram[i][j][k] = 255 * (histogram[i][j][k] - min) / delta;	//각 block을 255으로 normalization
			}

		}
	}

}
double GetRelation(int histogram1[][7][59], int histogram2[][7][59]) {
	int factor = 7 * 7 * 59;
	double Relation = 0.0;
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 59; k++) {
				Relation += fabs(histogram1[i][j][k] - histogram2[i][j][k]);	//eucladian distance말고 다른방법으로.....
			}																	//좀더 잡음에 강하도록			
		}
	}
	Relation /= factor;
	return Relation;

}



void main() {
	VideoCapture capture(0);
	Mat frame, Face_source;
	Mat Ref_Face;
	Mat Target;
	int start_flag = 0;


	CascadeClassifier cascade;
	cascade.load("C:\\Users\\illus\\Downloads\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml");

	printf("Select Method\n1 : Magnitude lbp\n2 : cosine similarity\n");
	scanf_s("%d", &method_flag);
	if (method_flag == 1 || method_flag == 2) {

		while (true) {
			if (start_flag == 0) {
				capture >> frame;
				imshow("cam", frame);

				int key = waitKey(30);
				if (key == 27)break;
				if (key == 'p' || key == 'P') {

					Face_source = frame.clone();

					vector<Rect>face;
					cascade.detectMultiScale(Face_source, face, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));
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



						start_flag = 1;
						destroyAllWindows();
						imshow("ref", Face_source);
						waitKey(30);

					}
					else {
						continue;
					}

				}
			}
			else {
				capture >> Target;

				vector<Rect> face_tar;
				cascade.detectMultiScale(Target, face_tar, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));
				if (face_tar.size() > 0) {
					for (int y = 0; y < face_tar.size(); y++) {
						int Tar_Histogram[7][7][59] = { 0 };
						Point lb(face_tar[y].x + face_tar[y].width, face_tar[y].y + face_tar[y].height);
						Point tr(face_tar[y].x, face_tar[y].y);
						Mat Tar_Face;
						Tar_Face = Target(Range(face_tar[y].y, face_tar[y].y + face_tar[y].height), Range(face_tar[y].x, face_tar[y].x + face_tar[y].width));

						Mat Tar_Face_gray;
						Mat Tar_LBP(Tar_Face.rows, Tar_Face.cols, CV_8UC1);
						cvtColor(Tar_Face, Tar_Face_gray, COLOR_BGR2GRAY);
						LBP(Tar_Face_gray, Tar_LBP);
						GetHist(Tar_LBP, Tar_Histogram);

						double relation = GetRelation(Ref_Histogram, Tar_Histogram);
						if (method_flag == 1) {
							if (relation < 5.0) {
								rectangle(Target, lb, tr, Scalar(0, 255, 0), 2, 8, 0);
							}
							else {
								rectangle(Target, lb, tr, Scalar(0, 0, 255), 2, 8, 0);
							}
						}
						else if (method_flag == 2) {
							if (relation < 0.9) {
								rectangle(Target, lb, tr, Scalar(0, 255, 0), 2, 8, 0);
							}
							else {
								rectangle(Target, lb, tr, Scalar(0, 0, 255), 2, 8, 0);
							}
						}
					
					}
				}
				else {
					continue;
				}
				imshow("matching", Target);
				int key = waitKey(30);
				if (key == 27)break;

			}

		}
	}
}