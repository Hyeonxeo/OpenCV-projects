#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

//Resizing image

int main() {
	Mat img_src = imread("test3.jpg", IMREAD_COLOR);
	float ScaleFactor;
	printf("scale factor : ");
	scanf("%f", &ScaleFactor);

	int SRCheight = img_src.rows, SRCwidth = img_src.cols;
	int REheight=SRCheight*ScaleFactor, REwidth=SRCwidth * ScaleFactor;

	Mat img_result(REheight, REwidth, CV_8UC3);
	int y, x,i;
	float ReverseFactor = 1 / ScaleFactor;

	if (ScaleFactor < 1) {
		for (y = 0; y < REheight; y++) {
			for (x = 0; x < REwidth; x++) {
				int Y = (y * ReverseFactor) / 1, X = (x * ReverseFactor) / 1;
				uchar p1, p2, p3, p4;

				for (i = 0; i < 3; i++) {
					p1 = img_src.at<Vec3b>(Y, X)[i];
					p2 = img_src.at<Vec3b>(Y, X + 1)[i];
					p3 = img_src.at<Vec3b>(Y + 1, X)[i];
					p4 = img_src.at<Vec3b>(Y + 1, X + 1)[i];
					float del = 1 - ScaleFactor;
					float sum = (del * del * p1) + (del * ScaleFactor * p2) + (del * ScaleFactor * p3) + (ScaleFactor * ScaleFactor * p4);

					img_result.at<Vec3b>(y, x)[i] = (uchar)sum;
				}
			}
		}
	}
	else if (ScaleFactor > 1) {
		for (y = 0; y < REheight; y++) {
			for (x = 0; x < REwidth; x++) {
				int Y = (y * ReverseFactor) / 1, X = (x * ReverseFactor) / 1;
				uchar p1, p2, p3, p4;
				if (X >= 0 && X+1 < SRCwidth && Y >= 0 && Y+1 < SRCheight) {
					for (i = 0; i < 3; i++) {
						p1 = img_src.at<Vec3b>(Y, X)[i];
						p2 = img_src.at<Vec3b>(Y, X + 1)[i];
						p3 = img_src.at<Vec3b>(Y + 1, X)[i];
						p4 = img_src.at<Vec3b>(Y + 1, X + 1)[i];
						float del = 1 - ReverseFactor;
						float sum = (del * del * p1) + (del * ReverseFactor * p2) + (del * ReverseFactor * p3) + (ReverseFactor * ReverseFactor * p4);

						img_result.at<Vec3b>(y, x)[i] = (uchar)sum;
					}
				}

			}
		}
	}
	else {
		img_result = img_src.clone();
	}
	imshow("source", img_src);
	imshow("result", img_result);

	cv::waitKey(0);
	return 0;
}