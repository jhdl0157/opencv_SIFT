

#include<opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>  
#include <opencv2/imgcodecs.hpp>  
#include <opencv2/imgproc.hpp>  
#include <opencv2/features2d.hpp>
#include <stdio.h>
#include "opencv2/calib3d.hpp"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;


int main()

{
	Mat ImageL;
	Mat lmageR;
	Mat GrayL;
	Mat GrayR;

	ImageL = imread("leftt.jpg", IMREAD_COLOR);
	lmageR = imread("rightt.jpg", IMREAD_COLOR);

	// 이미지 전처리
	Size size(ImageL.cols / 8, ImageL.rows / 8);
	resize(ImageL, ImageL, size);
	resize(lmageR, lmageR, size);
	cvtColor(ImageL, GrayL, COLOR_BGR2GRAY);
	cvtColor(lmageR, GrayR, COLOR_BGR2GRAY);

	//SIFT이용하기 
	double Thresold  = 500.;  
	Ptr<SiftFeatureDetector> Detector = SIFT::create(Thresold);
	vector< KeyPoint > vecL, vecR;

	//특징값 벡터에 저장 
	Detector->detect(GrayL, vecL);
	Detector->detect(GrayR, vecR);

	//Descriptors 구하기 
	Ptr<SiftDescriptorExtractor>Extractor = SIFT::create();
	Mat Descriptors1, Descriptors2;
	Extractor->compute(GrayL, vecL, Descriptors1);
	Extractor->compute(GrayR, vecR, Descriptors2);

	//Flann매칭 시도
	FlannBasedMatcher Matcher;
	vector<DMatch> matches;
	Matcher.match(Descriptors1, Descriptors2, matches);

	//점사이 거리 구하기 
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;
	for (int i = 0; i < Descriptors1.rows; i++)
	{
		dDistance = matches[i].distance;
		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;
	}
	// good매치만 구하기 
	vector<DMatch> good_matches;
	for (int i = 0; i < Descriptors1.rows; i++)
	{
		if (matches[i].distance < 5 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	Mat Goodmatch;
	drawMatches(ImageL, vecL, lmageR, vecR, good_matches, Goodmatch, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	imshow("good-matches", Goodmatch);
	
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(vecL[good_matches[i].queryIdx].pt);
		scene.push_back(vecR[good_matches[i].trainIdx].pt);
	}
	imwrite("sift_goodmatch.jpg", Goodmatch);
	//호모그래피 구하기 
	Mat H = findHomography(scene, obj,RANSAC);
	cout << H << endl;

	// homoG를 이용한 warp구하기 
	Mat temp;
	warpPerspective(lmageR, temp, H, Size(lmageR.cols *2, lmageR.rows), INTER_CUBIC);
	Mat result;
	result = temp.clone();
	printf("WARP된 이미지를 저장하였습니다.\n");
	imwrite("sift_warp.jpg", result);

	Mat matROI(result, Rect(0, 0, ImageL.cols, ImageL.rows));
	ImageL.copyTo(matROI);
	printf("합쳐진 이미지를 저장하였습니다.");
	imshow("result_image", result);
	waitKey(0);
	imwrite("result_image.jpg", result);

	return 0;

}


