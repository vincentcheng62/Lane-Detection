#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#define Need_Rotate_R (1)
using namespace cv;
class CalibrationRecord
{
public:
	CalibrationRecord() {}
	CalibrationRecord(char* filename, bool isVerbose = false) : isVerbose_(isVerbose)
	{
		if (isVerbose_) printf("[Calibration result]\n");
		cv::FileStorage fs(filename, FileStorage::READ);
		fs["Mint"] >> Mint;
		if (isVerbose_) std::cout << "Mint: " << Mint << std::endl;
		fs["Mext"] >> Mext;
		if (Need_Rotate_R) Mext.colRange(0, 3) = Mext.colRange(0, 3).t();
		if (isVerbose_) std::cout << "Mext: " << Mext << std::endl;
		fs["distCoeffs"] >> distCoeffs;
		if (isVerbose_) std::cout << "distCoeffs: " << distCoeffs.t() << std::endl;
		fs.release();

		cv::Rodrigues(Mext.colRange(0, 3), rotation_vec);
		if (isVerbose_) std::cout << "Rotation vector: " << rotation_vec.t() << std::endl;
		if (isVerbose_) std::cout << "Rotation vector (in degree): " << rotation_vec.t()*180/CV_PI << std::endl;

		cv::Mat Worldzeroatimage = Mint*Mext.col(3)*(1 / Mext.at<double>(2, 3));
		WorldcenterAtImg = cv::Point2f(Worldzeroatimage.at<double>(0), Worldzeroatimage.at<double>(1));
		if (isVerbose_) printf("world zero on image: %f, %f\n", WorldcenterAtImg.x, WorldcenterAtImg.y);

		ground_dist = sqrt(pow(Mext.at<double>(1, 3), 2) + pow(Mext.at<double>(2, 3), 2))*cos(CV_PI*0.5f + rotation_vec.at<double>(0));
		if (isVerbose_) std::cout << "World ground-displacement between camera and world zero: " << ground_dist << std::endl;

		Norm_of_T = sqrt(pow(Mext.at<double>(0, 3), 2) + pow(ground_dist, 2));
		if (isVerbose_) std::cout << "Total distance between camera and world zero (i.e. Norm2(T)): " << Norm_of_T << std::endl;

		orig_angle = atan(Mext.at<double>(0, 3) / ground_dist);
		if (isVerbose_) std::cout << "Angle between World zero and camera in x-direction " << orig_angle << "/ " << orig_angle*180/CV_PI << "degree" << std::endl;
		if (isVerbose_) std::cout << std::endl;
	}

	void CalcWrappingMatrix(cv::Size sz) // size of the img from video stream
	{
		if (isVerbose_) printf("[Wrapping result]\n");
		double diagonal = sqrt(sz.width*sz.width + sz.height*sz.height);
		if (isVerbose_)std::cout << "diagonal length of img ROI: " << diagonal << std::endl;

		cv::Mat obj_corners = cv::Mat::ones(3, 5, Mint.type());
		cv::Mat scene_corners = cv::Mat::zeros(3, 5, Mint.type());
		obj_corners.at<double>(0, 0) = 0;
		obj_corners.at<double>(1, 0) = 0;
		obj_corners.at<double>(0, 1) = sz.width;
		obj_corners.at<double>(1, 1) = 0;
		obj_corners.at<double>(0, 2) = 0;
		obj_corners.at<double>(1, 2) = sz.height;
		obj_corners.at<double>(0, 3) = sz.width;
		obj_corners.at<double>(1, 3) = sz.height;
		obj_corners.at<double>(0, 4) = WorldcenterAtImg.x; // temp.cols*0.5f;
		obj_corners.at<double>(1, 4) = WorldcenterAtImg.y; //temp.rows*0.5f;
		//if (isVerbose_)printf("optical center: %f, %f\n", obj_corners.at<double>(0, 4), obj_corners.at<double>(1, 4));
		if (isVerbose_)printf("image center: %f, %f\n", sz.width*0.5f, sz.height*0.5f);

		persT = Mext.colRange(0, 2);
		cv::hconcat(persT, Mext.col(3), persT);
		persT = (Mint*persT).inv();
		scene_corners = persT*obj_corners;

		double minx = 1e10, miny = 1e10, maxx = 0, maxy = 0;
		for (int i = 0; i < scene_corners.cols; i++)
		{
			scene_corners.at<double>(0, i) /= scene_corners.at<double>(2, i);
			scene_corners.at<double>(1, i) /= scene_corners.at<double>(2, i);
			if (scene_corners.at<double>(0, i) < minx) minx = scene_corners.at<double>(0, i);
			if (scene_corners.at<double>(0, i) > maxx) maxx = scene_corners.at<double>(0, i);
			if (scene_corners.at<double>(1, i) < miny) miny = scene_corners.at<double>(1, i);
			if (scene_corners.at<double>(1, i) > maxy) maxy = scene_corners.at<double>(1, i);
		}

		xsize = maxx - minx;
		ysize = maxy - miny;
		ratio = diagonal / min(xsize, ysize);
		if (isVerbose_)std::cout << "ratio of scale: " << ratio << std::endl;
		double xoffset = minx < 0 ? -1 * ratio*minx : 0;
		double yoffset = miny < 0 ? -1 * ratio*miny : 0;
		if (isVerbose_)printf("xoffset: %f, yoffset: %f\n", xoffset, yoffset);
		double xcenter = scene_corners.at<double>(0, 4)*ratio + xoffset;
		double ycenter = scene_corners.at<double>(1, 4)*ratio + yoffset;
		wrapped_World_center = cv::Point2f(xcenter, ycenter);
		if (isVerbose_)printf("wrapped world center in wrapped img space: %f, %f\n", xcenter, ycenter);
		if (isVerbose_)printf("Wrapped size: %f, %f\n", floor(xsize*ratio) + 1, floor(ysize*ratio) + 1);

		cv::Mat correctionmat = cv::Mat::eye(3, 3, Mint.type());
		correctionmat.at<double>(0, 0) = ratio;
		correctionmat.at<double>(1, 1) = ratio;
		correctionmat.at<double>(0, 2) = xoffset;
		correctionmat.at<double>(1, 2) = yoffset;

		persT = correctionmat*persT;
		if (isVerbose_)std::cout << "Correction matrix: " << correctionmat << std::endl;
		if (isVerbose_)std::cout << std::endl;
	}

	cv::Vec4f wrapline(cv::Vec4f line)
	{
		cv::Mat line_endpt = cv::Mat::ones(3, 2, Mint.type());
		cv::Mat line_endpt_wrap = cv::Mat::ones(3, 2, Mint.type());
		line_endpt.at<double>(0, 0) = line[0];
		line_endpt.at<double>(1, 0) = line[1];
		line_endpt.at<double>(0, 1) = line[2];
		line_endpt.at<double>(1, 1) = line[3];
		line_endpt_wrap = persT*line_endpt;

		return 	Vec4f(line_endpt_wrap.at<double>(0, 0) / line_endpt_wrap.at<double>(2, 0),
			line_endpt_wrap.at<double>(1, 0) / line_endpt_wrap.at<double>(2, 0),
			line_endpt_wrap.at<double>(0, 1) / line_endpt_wrap.at<double>(2, 1),
			line_endpt_wrap.at<double>(1, 1) / line_endpt_wrap.at<double>(2, 1));
	}

	cv::Mat Mint;
	cv::Mat Mext;
	cv::Mat distCoeffs;


	cv::Mat rotation_vec;
	cv::Point2f WorldcenterAtImg;
	double ground_dist;
	double Norm_of_T;
	double orig_angle;

	double ratio;
	cv::Point2f wrapped_World_center;
	cv::Mat persT;
	double xsize;
	double ysize;

	bool isVerbose_;
};