#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils.h"
#include "utils.cpp"
using namespace cv;
using namespace std;


// ORB settings
int ORB_MAX_KPTS = 1500;
float ORB_SCALE_FACTOR = 1.8;
int ORB_PYRAMID_LEVELS = 4;
float ORB_EDGE_THRESHOLD = 80.0;
int ORB_FIRST_PYRAMID_LEVEL = 0;
int ORB_WTA_K = 2;
int ORB_PATCH_SIZE = 31;
// Some image matching options
float MIN_H_ERROR = 2.50f; // Maximum error in pixels to accept an inlier
float DRATIO = 0.80f;

void main() {
	Mat img1, img1_32, img1_G, img2, img2_32, img2_G;
//	string img_path1, img_path2, homography_path;


	vector<KeyPoint> kpts1_orb;
	Mat desc1_orb;

	// Color images for results visualization
	Mat img_com_orb;

	//Video
	VideoCapture capture(0);
	int start_flag = 0;

	Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
		ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB::HARRIS_SCORE,
		ORB_PATCH_SIZE);

	// Create the L2 and L1 matchers
	Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
	Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");

	while (true) {

		if (start_flag == 0) {
			capture >> img1;
			imshow("img_com", img1);
			int key = waitKey(30);
			if (key == 27)break;
			if (key == 'p' || key == 'P') {
				start_flag = 1;
				img_com_orb = Mat(Size(img1.cols * 2, img1.rows), CV_8UC3);

				cvtColor(img1, img1_G, COLOR_BGR2GRAY);
				img1_G.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

				orb->detectAndCompute(img1_G, noArray(), kpts1_orb, desc1_orb, false);
				draw_keypoints(img1, kpts1_orb);
				destroyAllWindows();
			}
		}
		else {
			capture >> img2;
			cvtColor(img2, img2_G, COLOR_BGR2GRAY);
			// Convert the images to float			
			img2_G.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

			vector<KeyPoint>  kpts2_orb;
			vector<Point2f> matches_orb, inliers_orb;
			vector<vector<DMatch>> dmatches_orb;
			Mat  desc2_orb;

			orb->detectAndCompute(img2_G, noArray(), kpts2_orb, desc2_orb, false);
			matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
			matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);
			compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);

			draw_keypoints(img2, kpts2_orb);
			draw_inliers(img1, img2, img_com_orb, inliers_orb, 0);
			cv::imshow("ORB", img_com_orb);
    		waitKey(100);
		}

	}
}
