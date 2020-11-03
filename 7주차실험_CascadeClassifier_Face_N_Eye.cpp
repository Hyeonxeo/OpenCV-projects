#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;

void main() {
	Mat img = imread("testimage.jpg", IMREAD_COLOR);
	vector<Rect> faces, eyes;

	CascadeClassifier cascade,cascade_eye;
	cascade.load("C:\\Users\\illus\\Downloads\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml");
	cascade.detectMultiScale(img, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

	cascade_eye.load("C:/Users/illus/Downloads/opencv/sources/data/haarcascades/haarcascade_eye.xml");
	cascade_eye.detectMultiScale(img, eyes, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

	for (int y = 0; y < faces.size(); y++) {
		Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
		Point tr(faces[y].x, faces[y].y);

		rectangle(img, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
	}
	for (int y = 0; y < eyes.size(); y++) {
		Point lb(eyes[y].x + eyes[y].width, eyes[y].y + eyes[y].height);
		Point tr(eyes[y].x, eyes[y].y);

		rectangle(img, lb, tr, Scalar(255, 0, 0), 3, 8, 0);
	}
	
	imshow("result", img);
	waitKey(0);
}