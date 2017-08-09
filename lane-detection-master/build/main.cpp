#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "boost/filesystem.hpp"
#include <boost/algorithm/string.hpp>  
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <math.h> 
#include "geometry.h"
#include "LaneTrackingSystem.h"

#define PATH_TO_IMAGES "F:/lane_marking_detection/calibration_image/basler/video/video1"
#define Calibration_File "F:/lane_marking_detection/calibration_image/result_copy.yml"
#define Global_Map_File "F:/lane_marking_detection/calibration_image/global_map.yml"

//#define RATIO 0.5
#define LANE_PHYSICAL_WIDTH		(352)
#define LEFT_LANE_TO_WORLD_CENTER_XDIST	(117)
//#define LANE_PHYSICAL_WIDTH_THD		(15)
//#define PARALLEL_LINE_ANGLE_DIFF	((2*CV_PI)/180)

#define SKIP_FRAME_RATIO	(2)
//#define RESIZE_RATIO  (0.7)

using namespace cv;

std::vector<std::string> GetFileNamesInDirectory(const std::string directory)
{
	std::vector<std::string> files;
	try {
		boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
		for (boost::filesystem::directory_iterator i(directory); i != end_itr; ++i) {
			// Skip if not a file
			if (!boost::filesystem::is_regular_file(i->status()))
				continue;

			// File matches, store it
			std::string tmp = i->path().extension().string();
			boost::algorithm::to_lower(tmp);
			if (tmp == ".bmp")	files.push_back(i->path().string());
		}
	}
	catch (std::exception& e) { std::cerr << e.what() << std::endl; }
	return files;
}

int main(int argc, char* argv[])
{
	auto isVerbose = true;
	// Get all *.jpg in the GT luggage database and testcase database
	std::vector<std::string> vFileNames = GetFileNamesInDirectory(PATH_TO_IMAGES);
	std::sort(vFileNames.begin(), vFileNames.end());
	auto nImage = vFileNames.size();

	//Do the transformation matrix
	cv::Mat temp, tempgray, tempresize, tempimage = imread(vFileNames[0]);
	//cvtColor(tempimage, tempgray, CV_RGB2GRAY);
	//cv::resize(tempgray, tempresize, cv::Size(0, 0), RESIZE_RATIO, RESIZE_RATIO);

	//Initialize the lane tracking system
	LaneTrackingSystem LTS((char*)Calibration_File, tempimage.size(), 
		LANE_PHYSICAL_WIDTH, LEFT_LANE_TO_WORLD_CENTER_XDIST, isVerbose);
	LTS.LoadGlobalMap((char*)Global_Map_File);

	if(isVerbose) std::cout << "Number of images: " << nImage << std::endl;
	if(isVerbose) std::cout << "Start lane detection...\n" << std::endl;
	int64	stime, etime, stime2, etime2;
	//int quad = 0, sign = 0;

	for (int i = 0; i < nImage; i += SKIP_FRAME_RATIO)
	{
		stime = cv::getTickCount();
		cv::Mat resize, gray, image = imread(vFileNames[i]);
		cvtColor(image, gray, CV_RGB2GRAY);
		//cv::resize(gray, resize, cv::Size(0, 0), RESIZE_RATIO, RESIZE_RATIO);

		auto TrackResult = LTS.predict(gray);

		////undistort
		//stime2 = cv::getTickCount();
		//cv::Mat undistored;
		//undistort(gray, undistored, calib.Mint, calib.distCoeffs);
		//etime2 = cv::getTickCount();
		//printf("Undistort %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//// Canny algorithm
		//stime2 = cv::getTickCount();
		//cv::Mat blurred, canny, mask, maskerode;
		//cv::blur(undistored, blurred, cv::Size(BLUR_SIZE, BLUR_SIZE));
		//cv::Canny(blurred, canny, CANNYTHD, 2 * CANNYTHD);
		//cv::threshold(undistored, mask, 1, 255, THRESH_BINARY);
		//cv::Mat element = getStructuringElement(MORPH_RECT,Size(2 * 2 + 1, 2 * 2 + 1),Point(2, 2));
		//cv::erode(mask, maskerode, element);
		//canny = canny.mul(maskerode);
		//etime2 = cv::getTickCount();
		//printf("Filter and canny %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//stime2 = cv::getTickCount();
		//std::vector<Vec4i> lines;
		////HoughLines(contours, lines, 1, PI / 180, 200 * RATIO);
		//double minLineLength = canny.rows*0.3f;
		//double maxLineGap = canny.rows*0.3f;
		//HoughLinesP(canny, lines, 2, CV_PI / 90, HOUGHP_VOTE_THD, minLineLength, maxLineGap);
		//if (lines.size() == 0) misslinecounter++; else misslinecounter = 0;
		//etime2 = cv::getTickCount();
		//printf("Hough %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		////Wrap the line
		//std::vector<Vec4f> lines_wrap;
		//for (size_t j = 0; j < lines.size(); j++)
		//{
		//	lines_wrap.push_back(calib.wrapline(lines[j]));
		//	//cv::Mat line_endpt_wrap = calib.wrapline(line_endpt);
		//}

		////Wrapped
		//stime2 = cv::getTickCount();
		//cv::Mat wrapped;
		//if (isVerbose) warpPerspective(undistored, wrapped, calib.persT, cv::Size(floor(calib.xsize*calib.ratio) + 1,
		//	floor(calib.ysize*calib.ratio) + 1));

		//etime2 = cv::getTickCount();
		//if (isVerbose) printf("wrap %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//// Draw the lines in both wrapped and non-wrapped image and find the highest score line
		//cv::Mat houghresult = undistored.clone();
		//cv::Mat houghresultwrap = wrapped.clone();
		//int maxID = 0;
		//double maxscore = -1e10;
		//for (size_t j = 0; j < lines.size(); j++)
		//{
		//	cv::Point2f a(lines[j][0], lines[j][1]);
		//	cv::Point2f b(lines[j][2], lines[j][3]);
		//	line(houghresult, a, b, Scalar(0, 0, 255), 2, 8);

		//	cv::Point2f c(lines_wrap[j][0], lines_wrap[j][1]);
		//	cv::Point2f d(lines_wrap[j][2], lines_wrap[j][3]);
		//	if (isVerbose) line(houghresultwrap, c, d, Scalar(0, 0, 255), 2, 8);

		//	if (isVerbose) printf("line: %d, (%d, %d), (%d, %d)\n", j, lines[j][0], lines[j][1],
		//		lines[j][2], lines[j][3]);
		//	if (isVerbose) printf("linewrap: %d, (%f, %f), (%f, %f)\n", j, lines_wrap[j][0], lines_wrap[j][1],
		//		lines_wrap[j][2], lines_wrap[j][3]);

		//	double lengthscore = min(double(canny.rows*0.5f), LineLen(a, b));
		//	double dist_to_centre = Dist_Pt_To_Line(c, d, calib.wrapped_World_center);
		//	double final_score = lengthscore / dist_to_centre;

		//	if (isVerbose) printf("Lengthscore: %f, dist_to_centre: %f, final_score: %f\n", lengthscore,
		//		dist_to_centre, final_score);

		//	if (final_score > maxscore)
		//	{
		//		maxscore = final_score;
		//		maxID = j;
		//	}

		//}

		////Draw the best line in wrapped image
		//double rho=0.0f, theta = 0.0f;
		//cv::Point2f besta, bestb, besta_fit, bestb_fit;
		//cv::Mat bestline = wrapped.clone();
		//cv::Mat edgepoint = undistored.clone();
		//if (lines.size() > 0)
		//{
		//	cv::Point2f besta_beforewrap(lines[maxID][0], lines[maxID][1]);
		//	cv::Point2f bestb_beforewrap(lines[maxID][2], lines[maxID][3]);
		//	besta = cv::Point2f(lines_wrap[maxID][0], lines_wrap[maxID][1]);
		//	bestb = cv::Point2f(lines_wrap[maxID][2], lines_wrap[maxID][3]);
		//	cv::Rect ROI(besta_beforewrap, bestb_beforewrap);
		//	ROI.width += 1; ROI.height += 1;

		//	//Refine the line by WLS
		//	stime2 = cv::getTickCount();
		//	cv::Mat roi_raw = undistored(ROI);
		//	cv::Mat LooseMaskwls = cv::Mat::zeros(ROI.size(), wrapped.type());
		//	cv::Mat Maskwls = cv::Mat::zeros(ROI.size(), wrapped.type());
		//	line(LooseMaskwls, besta_beforewrap - cv::Point2f(ROI.tl()), bestb_beforewrap - cv::Point2f(ROI.tl()), Scalar(1), 13, 8);
		//	line(Maskwls, besta_beforewrap - cv::Point2f(ROI.tl()), bestb_beforewrap - cv::Point2f(ROI.tl()), Scalar(1), 2, 8);

		//	std::vector<cv::Point2f> Points;
		//	cv::Vec4f line;
		//	EdgesSubPix(roi_raw.mul(LooseMaskwls), Maskwls, Points, 0.7*CANNYTHD, 0.7*2 * CANNYTHD);
		//	etime2 = cv::getTickCount();
		//	//printf("Find subpixel edge %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//	if (Points.size() > 0)
		//	{
		//		stime2 = cv::getTickCount();
		//		cv::fitLine(Points, line, CV_DIST_L2, 0, 0.01, 0.01);
		//		etime2 = cv::getTickCount();
		//		printf("FitLine %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//		cv::Vec4f lineINorgROI(line[2] + ROI.x, line[3] + ROI.y, line[2] + ROI.x + 2500 * line[0],
		//			line[3] + ROI.y + 2500 * line[1]);
		//		cv::Vec4f Wrapped_bestline_afterfit = calib.wrapline(lineINorgROI);
		//		besta_fit = cv::Point2f(Wrapped_bestline_afterfit[0], Wrapped_bestline_afterfit[1]);
		//		bestb_fit = cv::Point2f(Wrapped_bestline_afterfit[2], Wrapped_bestline_afterfit[3]);

		//		//Draw the subpixel edge points found
		//		if (isVerbose)
		//		{
		//			for (int j = 0; j < Points.size(); j++)
		//			{
		//				circle(edgepoint, Points[j] + cv::Point2f(ROI.tl()), 2, Scalar(255), 1);
		//			}
		//		}



		//		if (isVerbose) cv::line(bestline, besta_fit, bestb_fit, Scalar(0, 0, 255), 2, 8);
		//		//Line_To_Polar_Coord(besta_fit, bestb_fit, rho, theta);
		//	}
		//	else 
		//	{
		//		besta_fit = besta; bestb_fit = bestb;
		//		if (isVerbose) printf("Fitting fail since no line is found!\n");
		//	}

		//	Line_To_Polar_Coord(besta_fit, bestb_fit, rho, theta);
		//	if (isVerbose) printf("best line: (%f, %f), (%f, %f)\n", besta_fit.x, besta_fit.y,
		//		bestb_fit.x, bestb_fit.y);
		//}


		////Kalman filter to track the best line
		//double precTick = ticks;
		//ticks = (double)cv::getTickCount();
		//double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
		//if (isVerbose) std::cout << "dT: " << dT << std::endl;
		//cv::Mat filterresult = wrapped.clone();
		//cv::Mat estimated;

		//if (filterFirsttime == true)
		//{
		//	if (lines.size() > 0 && dT < 0.5)
		//	{
		//		kf.statePre.at<double>(0) = rho;
		//		kf.statePre.at<double>(1) = 0.0f;
		//		kf.statePre.at<double>(2) = theta; // quad_wrap(theta);
		//		kf.statePre.at<double>(3) = 0.0f;
		//		kf.statePre.at<double>(4) = Dist_Pt_To_Line(besta, bestb, calib.wrapped_World_center);
		//		kf.statePre.at<double>(5) = 0.0f;

		//		kf.statePost = kf.statePre.clone();
		//		filterFirsttime = false;
		//		if (isVerbose) std::cout << "Kalman initialized at: " << kf.statePre.t() << std::endl;
		//	}

		//}
		//else
		//{
		//	stime2 = cv::getTickCount();

		//	kf.transitionMatrix.at<double>(1) = dT;
		//	kf.transitionMatrix.at<double>(15) = dT;
		//	kf.transitionMatrix.at<double>(29) = dT;

		//	cv::Mat prediction = kf.predict();
		//	if (isVerbose) std::cout << "prediction: " << prediction.t() << std::endl;

		//	//Update measurement
		//	if (lines.size() > 0)
		//	{
		//		measurement.at<double>(0) = rho;
		//		measurement.at<double>(1) = theta; // quad_wrap(theta);
		//		measurement.at<double>(2) = Dist_Pt_To_Line(besta, bestb, calib.wrapped_World_center);

		//		estimated = kf.correct(measurement);
		//		if (isVerbose) std::cout << "estimated: " << estimated.t() << std::endl;
		//		if (isVerbose) printf("Measurement: %f, %f, %f\n", rho, theta, Dist_Pt_To_Line(besta, bestb, calib.wrapped_World_center));
		//		//quad = floor((theta<0?theta+CV_PI:theta) / (CV_PI*0.5f));
		//		//sign = rho >= 0 ? 1 : -1;
		//	}
		//	else
		//	{
		//		estimated = prediction.clone();
		//		//sign1 and sign2 keep using last round result
		//	}

		//	if (misslinecounter < MAX_MISSING_LINE_FRAME && !estimated.empty() && isVerbose)
		//	{
		//		cv::Point2f pt1(estimated.at<double>(0) / cos(estimated.at<double>(2)), 0.0f);
		//		cv::Point2f pt2((estimated.at<double>(0) - filterresult.rows*
		//			sin(estimated.at<double>(2))) / cos(estimated.at<double>(2)),
		//			filterresult.rows*1.0f);
		//		line(filterresult, pt1, pt2, Scalar(255), 2);
		//		printf("(%d, %d), (%d, %d)\n", pt1.x, pt1.y, pt2.x, pt2.y);

		//	}

		//}


		//etime2 = cv::getTickCount();
		//printf("Kalman filter %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		////Make the bird-view display to show the AGV possible yaw angle and x-displacement
		//cv::Mat birdviewdisplay = cv::Mat::zeros(300, 300, CV_8UC3);
		//if (misslinecounter < MAX_MISSING_LINE_FRAME && !filterFirsttime && !estimated.empty())
		//{
		//	double xdelta_nonideal = estimated.at<double>(4) / calib.ratio;
		//	double yaw_angle = (estimated.at<double>(2) - calib.orig_angle);
		//	if (isVerbose) printf("xdelta_nonideal: %f\n", xdelta_nonideal);
		//	//double xdelta_ideal = LANE_PHYSICAL_WIDTH*0.5f - Mext.at<double>(0, 3) + y_dist*sin(estimated.at<double>(2)) -
		//	//	xdelta_nonideal / cos(estimated.at<double>(2));

		//	double xdelta_ideal = xdelta_nonideal -
		//		LEFT_LANE_TO_WORLD_CENTER_XDIST - calib.Norm_of_T*sin(yaw_angle) - calib.Mext.at<double>(0, 3);
		//	double AGV_xcoord_from_left_line = LEFT_LANE_TO_WORLD_CENTER_XDIST + -calib.Mext.at<double>(0, 3) + xdelta_ideal;

		//}


		char key = (char)waitKey(10);
		if (key == 'q')	break; 
	}
}
