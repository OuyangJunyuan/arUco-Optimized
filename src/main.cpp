#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <chrono>
#include <queue>
using namespace cv;
using namespace std;

bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();
    return true;
}

bool ArucoMarkerTraceOne(Mat &input,int id,std::vector<cv::Point2f> &corner,std::vector<cv::Point2f> &outputcorner)
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> param = cv::aruco::DetectorParameters::create();
    std::vector<std::vector<cv::Point2f>> candidates,MarkerCorners;
    std::vector<int> Ids;

    Rect max=Rect(0,0,0,0)+input.size(),temp = boundingRect(corner),roi_rect;
    roi_rect =  (temp - Point(temp.size().width,temp.size().height) + Size(temp.size()*2))&max;
    Mat roi=input(roi_rect);
    imshow("roi",roi);
    cv::aruco::detectMarkers(roi, dictionary, MarkerCorners, Ids, param, candidates);
    for (int i = 0; i < Ids.size(); ++i)
    {
        if(Ids[i]==id)
        {
            for (int j = 0; j < 4; ++j)
            {
                outputcorner[j] = Point2f(roi_rect.tl()) + MarkerCorners[i][j];
            }
            return true;
        }
    }
    return false;

}
void MakerDetect(Mat &inputImage,std::vector<cv::Vec3d> &rvecs, std::vector<cv::Vec3d> &tvecs,
                          Mat &camMatrix, Mat &distCoeffs,int id) {
    static int count = 0;
    static cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    static cv::Ptr<cv::aruco::DetectorParameters> param = cv::aruco::DetectorParameters::create();
    static std::vector<std::vector<cv::Point2f>> candidates, MarkerCorners;
    static std::vector<int> Ids;
    static bool istrace = false;

    if (count++%10==0) {
        cv::aruco::detectMarkers(inputImage, dictionary, MarkerCorners, Ids, param, candidates);
        for (int i = 0; i < Ids.size(); i++) {
            //if(Ids[i]==id)
                istrace=true;
        }
    }else{
        for (int i = 0; i < Ids.size(); i++) {
            //if(Ids[i]==id)
                istrace = ArucoMarkerTraceOne(inputImage, Ids[i], MarkerCorners[i], MarkerCorners[i]);
        }
    }
    if(istrace)
    {
        cv::aruco::estimatePoseSingleMarkers(MarkerCorners, 0.05, camMatrix, distCoeffs, rvecs, tvecs);
        for (int i = 0; i < Ids.size(); i++)
            cv::aruco::drawAxis(inputImage, camMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
    }

}

int main() {
    Mat camMatrix,distCoeffs;
    readCameraParameters("./../config/ost.yaml",camMatrix,distCoeffs);

    VideoCapture capture;
    capture.open(0);
    Mat frame,inputImage;

    int count=0;
    double timeCost=0;
    String dispString;
    while(capture.isOpened())
    {
        capture.read(frame);
        frame.copyTo(inputImage);


        std::vector<cv::Vec3d> rvecs,tvecs;
        chrono::steady_clock::time_point t1=chrono::steady_clock::now();
        /* ↓↓↓ estimate timecost ↓↓↓ */


        MakerDetect(inputImage,rvecs,tvecs,camMatrix,distCoeffs,23);

        /* ↑↑↑ estimate timecost ↑↑↑ */
        timeCost+=1000*(chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now()-t1).count());

        if(count++==100)
        {
            timeCost/=count;
            dispString=to_string(timeCost)+"ms";
            count=0;
            timeCost=0;
        }
        putText(inputImage,dispString , cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
        imshow("detect",inputImage);
        if(waitKey(1)=='q')
            break;
    }
    return 0;
}
