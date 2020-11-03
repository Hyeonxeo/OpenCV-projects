#include <opencv2/opencv.hpp>
#include <math.h>
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


int main() {
	Mat Img_Src = imread("test2.jpg", IMREAD_GRAYSCALE); //grayscale로 읽어서 filter에 넣을 것
	int height = Img_Src.rows;
	int width = Img_Src.cols;
	Mat Img_Filtered(height,width,CV_8UC1);
	LowPassFilter(Img_Src, Img_Filtered);
	int y, x, i,count;
	Mat Img_Result = imread("test2.jpg", IMREAD_COLOR);

	double theta = 22.5 * PI / 180;
	Scalar c;
	Point pCenter;
	int radius = 1;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			count = 0;
			int flag = 0;
			for (i = 0; i < 16; i++) {
				int dY =  (-3 * cos(theta*i));
				int dX =  (3 * sin(theta*i));
				int Y = y+dY;
				int X = x+dX;
				if (Y >= 0 && Y < height && X >= 0 && X < width) {
					if (Img_Src.at<uchar>(y, x) != Img_Src.at<uchar>(Y, X)) {
						count++;
					}
					else {
						count = 0;
					}

					if (count == 15) {
						flag = 1;			//연속된게 9가 되면~~
					}
				}
			}

			if (flag == 1) {
				pCenter.x = x;
				pCenter.y = y;
				c.val[0] = 0;
				c.val[1] = 0;
				c.val[2] = 255;
				circle(Img_Result, pCenter, radius, c, 1, 8, 0);
			}

		}
	}
	imshow("result", Img_Result);
	waitKey(0);

}