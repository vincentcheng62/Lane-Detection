#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "CalibrationRecord.h"
#include <iostream>

using namespace cv;

void cvShowManyImages(char* title, std::vector<std::string> namelist, int nArgs, ...);
void EdgesSubPix(cv::Mat gray, cv::Mat mask, std::vector<cv::Point2f> &Points, int low = 40,
	int high = 100, double alpha = 1.0f);

struct LaneTrackingResult {
	double yaw_angle=0;
	double x_delta=0;
	double x_coord=0; // real world x-dist from left lane in mm
	bool status=0;
};

class LaneTrackingSystem
{
public:
	cv::KalmanFilter InitKalmanFilter()
	{
		//Setup required matrix for kalman filter
		int stateSize = 6;
		int measSize = 3;
		int contrSize = 0;

		cv::KalmanFilter kf(stateSize, measSize, contrSize, CV_64F);
		cv::setIdentity(kf.transitionMatrix);
		kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_64F);
		kf.measurementMatrix.at<double>(0) = 1.0f;
		kf.measurementMatrix.at<double>(8) = 1.0f;
		kf.measurementMatrix.at<double>(16) = 1.0f;

		cv::setIdentity(kf.processNoiseCov, Scalar::all(1e-1));
		cv::setIdentity(kf.measurementNoiseCov, Scalar::all(1e-3));
		cv::setIdentity(kf.errorCovPost, Scalar::all(1e-2));

		return kf;
	}

	void Draw_Dashed_line(cv::Mat img, cv::Point2f p1, cv::Point2f p2, int gap = 5)
	{
		LineIterator it(img, p1, p2, 8);            // get a line iterator
		for (int i = 0; i < it.count; i++, it++)
			if (i % gap != 0) { (*it)[0] = 200; }
	}

	void construct_birdview(cv::Mat &birdviewdisplay, LaneTrackingResult result,
		double land_physical_width)
	{

		cv::Point2f x0(birdviewdisplay.cols * 2 / 3, 0);
		cv::Point2f x1(birdviewdisplay.cols * 2 / 3, birdviewdisplay.rows);
		cv::Point2f x2(birdviewdisplay.cols * 1 / 3, 0);
		cv::Point2f x3(birdviewdisplay.cols * 1 / 3, birdviewdisplay.rows);
		cv::Point2f y0(birdviewdisplay.cols * 0.5f, 0);
		cv::Point2f y1(birdviewdisplay.cols * 0.5f, birdviewdisplay.rows);

		double lanewidthratio = (birdviewdisplay.cols / 3.0f) / land_physical_width;


		cv::Point2f AGV(birdviewdisplay.cols * 1 / 3 + lanewidthratio*result.x_coord,
			birdviewdisplay.rows * 2 / 3);
		cv::Point2f AGVright(birdviewdisplay.cols * 2 / 3, AGV.y);
		cv::Point2f yawpoint(AGV.x - 50 * sin(result.yaw_angle), AGV.y - 50 * cos(result.yaw_angle));
		cv::Point2f arrow1(yawpoint.x, yawpoint.y);

		line(birdviewdisplay, x0, x1, Scalar(255, 255, 233), 3);
		line(birdviewdisplay, x2, x3, Scalar(255, 255, 233), 3);
		arrowedLine(birdviewdisplay, AGV, yawpoint, Scalar(0, 0, 255), 1);
		Draw_Dashed_line(birdviewdisplay, y0, y1, 5);
		circle(birdviewdisplay, AGV, 6, Scalar(0, 0, 255), 2);
		line(birdviewdisplay, AGV, AGVright, Scalar(0, 255, 255), 1);

		std::stringstream stream2;
		stream2 << "Yaw angle: " << result.yaw_angle * 180 / CV_PI << "deg";
		putText(birdviewdisplay, stream2.str(), Point(10, birdviewdisplay.rows - 40), 2, 0.3, Scalar(0, 255, 0), 0);
		std::stringstream stream3;
		stream3 << "x-delta:  " << result.x_delta << "mm";
		putText(birdviewdisplay, stream3.str(), Point(10, birdviewdisplay.rows - 20), 2, 0.3, Scalar(0, 255, 0), 0);

	}

	//The x-displacement (right is +ve) moving from angle orig to angle orig+yaw
	//Angle defined clockwisely, up is 0
	double Intermediate_arc_xdisplacement(double radius, double yaw, double orig)
	{
		double xdist_orig = radius*sin(orig);
		double xdist_orig_with_yaw = radius*sin(orig + yaw);
		return xdist_orig_with_yaw - xdist_orig;
	}

	LaneTrackingSystem(char* Calibrationfilename, cv::Size sz, double lanephysicalwidth,
		double leftlane_to_world_phyxdist, bool isVerbose = false) :
		lanephysicalwidth_(lanephysicalwidth), leftlane_to_world_phyxdist_(leftlane_to_world_phyxdist),
		isVerbose_(isVerbose)
	{
		calib = CalibrationRecord(Calibrationfilename, isVerbose);
		calib.CalcWrappingMatrix(sz);
		kf = InitKalmanFilter();
		//measurement = cv::Mat::create(kf.measurementMatrix.rows, 1, kf.measurementMatrix.type());
		if (isVerbose) printf("LaneTrackingSystem finish initialization.\n");
	}

	//Format of global map
	//Points: Each row is a cv::Vec3f containing a 3D point
	//Lines: Each row is a line containing the start and end point index with reference to Points 
	void LoadGlobalMap(char* filename)
	{
		cv::FileStorage fs(filename, FileStorage::READ);
		fs["points"] >> globalmap_points;
		if (isVerbose_) std::cout << "Global map points: " << globalmap_points << std::endl;
		fs["lines"] >> globalmap_lines;
		if (isVerbose_) std::cout << "Global map lines: " << globalmap_lines << std::endl;
		printf("Finish reading global map...\n");
	}

	//According to the given pose, find the most possible left pane to the AGV in the global map
	//Output the refine pose of the AGV
	cv::Vec6f RefineWithKiloMeter(cv::Mat img, cv::Vec6f pose)
	{
		if (globalmap_points.empty()) throw std::invalid_argument("Global map is not loaded!!");

		cv::Vec6f refinedpose(pose);
		auto predict_result = predict(img);
		if (predict_result.status) {

			int minID = 0;
			double mindistance = 1e10;
			cv::Point2f AGV_2dloc(pose(0), pose(1));
			for (int i = 0; i < globalmap_lines.rows; i++)
			{
				int startid = globalmap_lines.at<int>(i, 0);
				int endid = globalmap_lines.at<int>(i, 1);

				cv::Point2f start(globalmap_points.at<double>(startid, 0), globalmap_points.at<double>(startid, 1));
				cv::Point2f end(globalmap_points.at<double>(endid, 0), globalmap_points.at<double>(endid, 1));
				auto temp = Dist_Pt_To_Line(start, end, AGV_2dloc);
				if (temp >= 0 && temp < mindistance)
				{
					mindistance = temp;
					minID = i;
				}
			}

			refinedpose(0) += predict_result.x_delta;
			refinedpose(3) += predict_result.yaw_angle;
		}
		return refinedpose;

	}

	LaneTrackingResult predict(cv::Mat img)
	{
		cv::Mat measurement(kf.measurementMatrix.rows, 1, kf.measurementMatrix.type());
		int64 stime, etime, stime2, etime2;
		stime = cv::getTickCount();
		//undistort
		stime2 = cv::getTickCount();
		cv::Mat undistored;
		undistort(img, undistored, calib.Mint, calib.distCoeffs);
		etime2 = cv::getTickCount();
		printf("Undistort %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		// Canny algorithm
		stime2 = cv::getTickCount();
		cv::Mat blurred, canny, mask, maskerode;
		cv::blur(undistored, blurred, cv::Size(blur_size, blur_size));
		cv::Canny(blurred, canny, cannythd, 2 * cannythd);
		cv::threshold(undistored, mask, 1, 255, THRESH_BINARY);
		cv::Mat element = getStructuringElement(MORPH_RECT, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
		cv::erode(mask, maskerode, element);
		canny = canny.mul(maskerode);
		etime2 = cv::getTickCount();
		printf("Filter and canny %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		stime2 = cv::getTickCount();
		std::vector<Vec4i> lines;
		//HoughLines(contours, lines, 1, PI / 180, 200 * RATIO);
		HoughLinesP(canny, lines, 2, CV_PI / 90, houghp_vote_thd, 
			minLineLengthratio*canny.rows, maxLineGapratio*canny.rows);
		if (lines.size() == 0) misslinecounter++; else misslinecounter = 0;
		etime2 = cv::getTickCount();
		printf("Hough %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//Wrap the line
		std::vector<Vec4f> lines_wrap;
		for (size_t j = 0; j < lines.size(); j++)
		{
			lines_wrap.push_back(calib.wrapline(lines[j]));
			//cv::Mat line_endpt_wrap = calib.wrapline(line_endpt);
		}

		//Wrapped
		stime2 = cv::getTickCount();
		cv::Mat wrapped;
		if (isVerbose_) warpPerspective(undistored, wrapped, calib.persT, cv::Size(floor(calib.xsize*calib.ratio) + 1,
			floor(calib.ysize*calib.ratio) + 1));

		etime2 = cv::getTickCount();
		if (isVerbose_) printf("wrap %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		// Draw the lines in both wrapped and non-wrapped image and find the highest score line
		cv::Mat houghresult = undistored.clone();
		cv::Mat houghresultwrap = wrapped.clone();
		int maxID = 0;
		double maxscore = -1e10;
		for (size_t j = 0; j < lines.size(); j++)
		{
			cv::Point2f a(lines[j][0], lines[j][1]);
			cv::Point2f b(lines[j][2], lines[j][3]);
			line(houghresult, a, b, Scalar(0, 0, 255), 2, 8);

			cv::Point2f c(lines_wrap[j][0], lines_wrap[j][1]);
			cv::Point2f d(lines_wrap[j][2], lines_wrap[j][3]);
			if (isVerbose_) line(houghresultwrap, c, d, Scalar(0, 0, 255), 2, 8);

			if (isVerbose_) printf("line: %d, (%d, %d), (%d, %d)\n", j, lines[j][0], lines[j][1],
				lines[j][2], lines[j][3]);
			if (isVerbose_) printf("linewrap: %d, (%f, %f), (%f, %f)\n", j, lines_wrap[j][0], lines_wrap[j][1],
				lines_wrap[j][2], lines_wrap[j][3]);

			double lengthscore = min(double(canny.rows*0.5f), LineLen(a, b));
			double dist_to_centre = Dist_Pt_To_Line(c, d, calib.wrapped_World_center);
			double final_score = lengthscore / dist_to_centre;

			if (isVerbose_) printf("Lengthscore: %f, dist_to_centre: %f, final_score: %f\n", lengthscore,
				dist_to_centre, final_score);

			if (final_score > maxscore)
			{
				maxscore = final_score;
				maxID = j;
			}

		}

		//Draw the best line in wrapped image
		double rho = 0.0f, theta = 0.0f;
		cv::Point2f besta, bestb, besta_fit, bestb_fit;
		cv::Mat bestline = wrapped.clone();
		cv::Mat edgepoint = undistored.clone();
		if (lines.size() > 0)
		{
			cv::Point2f besta_beforewrap(lines[maxID][0], lines[maxID][1]);
			cv::Point2f bestb_beforewrap(lines[maxID][2], lines[maxID][3]);
			besta = cv::Point2f(lines_wrap[maxID][0], lines_wrap[maxID][1]);
			bestb = cv::Point2f(lines_wrap[maxID][2], lines_wrap[maxID][3]);
			cv::Rect ROI(besta_beforewrap, bestb_beforewrap);
			ROI.width += 1; ROI.height += 1;

			//Refine the line by WLS
			stime2 = cv::getTickCount();
			cv::Mat roi_raw = undistored(ROI);
			cv::Mat LooseMaskwls = cv::Mat::zeros(ROI.size(), wrapped.type());
			cv::Mat Maskwls = cv::Mat::zeros(ROI.size(), wrapped.type());
			line(LooseMaskwls, besta_beforewrap - cv::Point2f(ROI.tl()), bestb_beforewrap - cv::Point2f(ROI.tl()), Scalar(1), 13, 8);
			line(Maskwls, besta_beforewrap - cv::Point2f(ROI.tl()), bestb_beforewrap - cv::Point2f(ROI.tl()), Scalar(1), 2, 8);

			std::vector<cv::Point2f> Points;
			cv::Vec4f line;
			EdgesSubPix(roi_raw.mul(LooseMaskwls), Maskwls, Points, 0.7*cannythd, 0.7 * 2 * cannythd);
			etime2 = cv::getTickCount();
			//printf("Find subpixel edge %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

			if (Points.size() > 0)
			{
				stime2 = cv::getTickCount();
				cv::fitLine(Points, line, CV_DIST_L2, 0, 0.01, 0.01);
				etime2 = cv::getTickCount();
				printf("FitLine %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

				cv::Vec4f lineINorgROI(line[2] + ROI.x, line[3] + ROI.y, line[2] + ROI.x + 2500 * line[0],
					line[3] + ROI.y + 2500 * line[1]);
				cv::Vec4f Wrapped_bestline_afterfit = calib.wrapline(lineINorgROI);
				besta_fit = cv::Point2f(Wrapped_bestline_afterfit[0], Wrapped_bestline_afterfit[1]);
				bestb_fit = cv::Point2f(Wrapped_bestline_afterfit[2], Wrapped_bestline_afterfit[3]);

				//Draw the subpixel edge points found
				if (isVerbose_)
				{
					for (int j = 0; j < Points.size(); j++)
					{
						circle(edgepoint, Points[j] + cv::Point2f(ROI.tl()), 2, Scalar(255), 1);
					}
				}



				if (isVerbose_) cv::line(bestline, besta_fit, bestb_fit, Scalar(0, 0, 255), 2, 8);
				//Line_To_Polar_Coord(besta_fit, bestb_fit, rho, theta);
			}
			else
			{
				besta_fit = besta; bestb_fit = bestb;
				if (isVerbose_) printf("Fitting fail since no line is found!\n");
			}

			Line_To_Polar_Coord(besta_fit, bestb_fit, rho, theta);
			if (isVerbose_) printf("best line: (%f, %f), (%f, %f)\n", besta_fit.x, besta_fit.y,
				bestb_fit.x, bestb_fit.y);
		}


		//Kalman filter to track the best line
		double precTick = ticks;
		ticks = (double)cv::getTickCount();
		double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
		if (isVerbose_) std::cout << "dT: " << dT << std::endl;
		cv::Mat filterresult = wrapped.clone();
		cv::Mat estimated;

		if (filterFirsttime == true)
		{
			if (lines.size() > 0 && dT < 0.5)
			{
				kf.statePre.at<double>(0) = rho;
				kf.statePre.at<double>(1) = 0.0f;
				kf.statePre.at<double>(2) = theta; // quad_wrap(theta);
				kf.statePre.at<double>(3) = 0.0f;
				kf.statePre.at<double>(4) = Dist_Pt_To_Line(besta, bestb, calib.wrapped_World_center);
				kf.statePre.at<double>(5) = 0.0f;

				kf.statePost = kf.statePre.clone();
				filterFirsttime = false;
				if (isVerbose_) std::cout << "Kalman initialized at: " << kf.statePre.t() << std::endl;
			}

		}
		else
		{
			stime2 = cv::getTickCount();

			kf.transitionMatrix.at<double>(1) = dT;
			kf.transitionMatrix.at<double>(15) = dT;
			kf.transitionMatrix.at<double>(29) = dT;

			cv::Mat prediction = kf.predict();
			if (isVerbose_) std::cout << "prediction: " << prediction.t() << std::endl;

			//Update measurement
			if (lines.size() > 0)
			{
				measurement.at<double>(0) = rho;
				measurement.at<double>(1) = theta; // quad_wrap(theta);
				measurement.at<double>(2) = Dist_Pt_To_Line(besta, bestb, calib.wrapped_World_center);

				estimated = kf.correct(measurement);
				if (isVerbose_) std::cout << "estimated: " << estimated.t() << std::endl;
				if (isVerbose_) printf("Measurement: %f, %f, %f\n", rho, theta, Dist_Pt_To_Line(besta, bestb, calib.wrapped_World_center));
				//quad = floor((theta<0?theta+CV_PI:theta) / (CV_PI*0.5f));
				//sign = rho >= 0 ? 1 : -1;
			}
			else
			{
				estimated = prediction.clone();
				//sign1 and sign2 keep using last round result
			}

			if (misslinecounter < max_missing_line_frame && !estimated.empty() && isVerbose_)
			{
				cv::Point2f pt1(estimated.at<double>(0) / cos(estimated.at<double>(2)), 0.0f);
				cv::Point2f pt2((estimated.at<double>(0) - filterresult.rows*
					sin(estimated.at<double>(2))) / cos(estimated.at<double>(2)),
					filterresult.rows*1.0f);
				line(filterresult, pt1, pt2, Scalar(255), 2);
				printf("(%d, %d), (%d, %d)\n", pt1.x, pt1.y, pt2.x, pt2.y);

			}

		}


		etime2 = cv::getTickCount();
		printf("Kalman filter %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//Make the bird-view display to show the AGV possible yaw angle and x-displacement
		LaneTrackingResult result;
		cv::Mat birdviewdisplay = cv::Mat::zeros(300, 300, CV_8UC3);
		if (misslinecounter < max_missing_line_frame && !filterFirsttime && !estimated.empty())
		{
			double xdelta_nonideal = estimated.at<double>(4) / calib.ratio;
			result.yaw_angle = estimated.at<double>(2); // (estimated.at<double>(2) - calib.orig_angle);
			if (isVerbose_) printf("leftlane_to_world_xdist_nonidealspace: %f\n", xdelta_nonideal);
			//double xdelta_ideal = LANE_PHYSICAL_WIDTH*0.5f - Mext.at<double>(0, 3) + y_dist*sin(estimated.at<double>(2)) -
			//	xdelta_nonideal / cos(estimated.at<double>(2));

			result.x_delta = xdelta_nonideal -
				leftlane_to_world_phyxdist_ - Intermediate_arc_xdisplacement(calib.Norm_of_T, -1*result.yaw_angle, calib.orig_angle);
			result.x_coord = leftlane_to_world_phyxdist_ - calib.Mext.at<double>(0, 3) + result.x_delta;
			result.status = true;
			construct_birdview(birdviewdisplay, result, lanephysicalwidth_);
		}

		// Display the result


		etime = cv::getTickCount();
		double timeused = (double)(etime - stime) / (double)cv::getTickFrequency();
		printf("Total time for a frame %f seconds\n\n", timeused);

		std::stringstream stream;
		//stream << "Lines Segments: " << lines.size() << "   FPS: " << 1.0 / timeused;
		stream << "FPS: " << 1.0 / timeused;

		//cv::Mat FPSimg = wrapped.clone();
		putText(birdviewdisplay, stream.str(), Point(30, 70), 2, 0.4, Scalar(255, 255, 255), 0);
		line(houghresult, cv::Point(calib.WorldcenterAtImg.x - 20, calib.WorldcenterAtImg.y),
			cv::Point(calib.WorldcenterAtImg.x + 20, calib.WorldcenterAtImg.y), Scalar(255), 2);
		line(houghresult, cv::Point(calib.WorldcenterAtImg.x, calib.WorldcenterAtImg.y - 20),
			cv::Point(calib.WorldcenterAtImg.x, calib.WorldcenterAtImg.y + 20), Scalar(255), 2);
		putText(houghresult, "World(0, 0)", calib.WorldcenterAtImg + cv::Point2f(5, 17), 2, 0.6, Scalar(255, 255, 255), 0);
		//Draw the center of projected scene coord
		if (isVerbose_)
		{
			double xcenter = calib.wrapped_World_center.x;
			double ycenter = calib.wrapped_World_center.y;
			line(bestline, cv::Point(xcenter - 20, ycenter), cv::Point(xcenter + 20, ycenter), Scalar(255), 2);
			line(bestline, cv::Point(xcenter, ycenter - 20), cv::Point(xcenter, ycenter + 20), Scalar(255), 2);
			line(houghresultwrap, cv::Point(xcenter - 20, ycenter), cv::Point(xcenter + 20, ycenter), Scalar(255), 2);
			line(houghresultwrap, cv::Point(xcenter, ycenter - 20), cv::Point(xcenter, ycenter + 20), Scalar(255), 2);
			line(filterresult, cv::Point(xcenter - 20, ycenter), cv::Point(xcenter + 20, ycenter), Scalar(255), 2);
			line(filterresult, cv::Point(xcenter, ycenter - 20), cv::Point(xcenter, ycenter + 20), Scalar(255), 2);
			//imshow("wrapped", FPSimg);
		}


		std::vector<std::string> namelist{ "Raw image", "Canny filter" ,"HoughLine","Bird view" ,
			"HoughLine in wrapped space","Best line with WLS fit", "Subpixel edge points",
			"Kalman filter best line" };

		if (isVerbose_)
		{
			cvShowManyImages("Result image", namelist, 8, undistored, canny, houghresult, birdviewdisplay,
				houghresultwrap, bestline, edgepoint, filterresult);
		}
		else
		{
			cvShowManyImages("Result image", namelist, 4, undistored, canny, houghresult, birdviewdisplay);
		}

		return result;
	}

	CalibrationRecord calib;
	KalmanFilter kf;
	//cv::Mat measurement;
	bool isVerbose_;
	double ticks = 0;
	bool filterFirsttime = true;
	int misslinecounter = 0;
	double lanephysicalwidth_;
	double leftlane_to_world_phyxdist_;

	int max_missing_line_frame = 4;
	int cannythd = 25;
	int blur_size = 7;
	int houghp_vote_thd = 80;
	double minLineLengthratio = 0.3f;
	double maxLineGapratio = 0.3f;

	cv::Mat globalmap_points;
	cv::Mat globalmap_lines;

};