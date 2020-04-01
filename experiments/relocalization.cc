#include<string>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;
using namespace ORB_SLAM2;

struct Point3D{
    float x,y,z;
    vector<int> imageIds;
    vector<int> points2dIds;
};

struct Point2D{
    float x,y,scale,orientation;
    // bool operator < (const Point2D& other) const{
    //     return scale > other.scale;
    // }
};

// bool compareKeyPoint(cv::KeyPoint k1, cv::KeyPoint k2){
//     return k1.octave < k2.octave;
// }

void readPoints3D(string filename, unordered_map<int, Point3D>& points3D){
    string line;
    ifstream points3DFile(filename);
    getline(points3DFile, line);
    getline(points3DFile, line);
    getline(points3DFile, line);
    int point3DId;
    float x,y,z;
    int imageId, point2DId;
    while(getline(points3DFile, line)){
        istringstream iss(line);
        iss >> point3DId;
        Point3D point3d;
        iss >> x >> y >> z;
        point3d.x = x;
        point3d.y = y;
        point3d.z = z;
        //ignore RGB and error
        iss >> x >> x >> x >> x;
        while(iss >> imageId){
            point3d.imageIds.push_back(imageId);
            iss >> point2DId;
            point3d.points2dIds.push_back(point2DId);
        }
        points3D[point3DId] = point3d;
    }
}

void readPoints2D(string filename, unordered_map<int, vector<Point2D>>& points2D){
    string line;
    ifstream points2DFile(filename);
    getline(points2DFile, line);
    int imageID, image2dId;
    float x,y,scale,orientation;
    while(getline(points2DFile, line)){
        istringstream iss(line);
        iss >> imageID;
        auto& point2ds = points2D[imageID];
        while(iss >> image2dId){
            iss >> x >> y >> scale >> orientation;
            point2ds.push_back({x,y,scale,orientation});
        }
    }
}

// void readPoints2D(string filename, unordered_map<int, vector<cv::KeyPoint>>& points2D){
//     string line;
//     ifstream points2DFile(filename);
//     getline(points2DFile, line);
//     int imageID, image2dId;
//     float x,y,scale,orientation;
//     while(getline(points2DFile, line)){
//         istringstream iss(line);
//         iss >> imageID;
//         auto& point2ds = points2D[imageID];
//         while(iss >> image2dId){
//             iss >> x >> y >> scale >> orientation;
//             // point2ds.push_back({x,y,scale,orientation});
//             int level = discretizeScale(scale, 1.2);
//             points2D.push_back(cv::KeyPoint(x,y,31,0,))
//         }
//     }
// }

void discretizeScale(float scaleColmap, float scaleFactor, int maxLevel, int& level, float&scale){
    level = 0;
    scale = 1;
    while(scaleColmap > scaleFactor && level < maxLevel-1){
        level++;
        scaleColmap /= scaleFactor;
        scale *= scaleFactor;
    }
}

void drawGraph(MapDrawer* mpMapDrawer, FrameDrawer* mpFrameDrawer)
{

    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",true,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);

    // Define Camera Render Object (for view / scene browsing)
    float mViewpointX = 0, mViewpointY = -0.7, mViewpointZ = -1.8;
    float mViewpointF = 500;
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ,0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    cv::namedWindow("ORB-SLAM2: Current Frame");

    bool bFollow = true;
    bool bLocalizationMode = false;

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        pangolin::FinishFrame();

        // cv::Mat im = mpFrameDrawer->DrawFrame();
        // cv::imshow("ORB-SLAM2: Current Frame",im);
        // cv::waitKey(1./30);

        if(menuReset)
        {
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            menuReset = false;
        }
    }

}

int main(int argc, char **argv)
{

    // ORBVocabulary* mpVocabulary = new ORBVocabulary();
    // cout << "Load Vocabulary File" << endl;
    // const string strVocFile("Vocabulary/ORBvoc.txt");
    // mpVocabulary->loadFromTextFile(strVocFile);

    // cout << "Create KeyFrame database and Map" << endl;
    // KeyFrameDatabase* mpKeyFrameDatabase = new KeyFrameDatabase(mpVocabulary);
    Map* pMap = new Map();

    string argo_img_files[] = {
        "experiments/ring_front_center/ring_front_center_315978411061152056.jpg", //image id 5
        "experiments/ring_front_center/ring_front_center_315978411527355224.jpg"  //image id 13
    };
    
    // cv::Mat Tcws[] = { cv::Mat::eye(4,4,CV_32F), cv::Mat::eye(4,4,CV_32F) };

    // Tcws[0].at<float>(0,0) = 0.99984074;
    // Tcws[0].at<float>(0,1) = 0.00870911;
    // Tcws[0].at<float>(0,2) = -0.0155791;
    // Tcws[0].at<float>(0,3) = -0.455713;
    // Tcws[0].at<float>(1,0) = -0.00889986;
    // Tcws[0].at<float>(1,1) = 0.99988574;
    // Tcws[0].at<float>(1,2) = -0.01221689;
    // Tcws[0].at<float>(1,3) = 0.0478371;
    // Tcws[0].at<float>(2,0) = 0.01547092;
    // Tcws[0].at<float>(2,1) = 0.0123536;
    // Tcws[0].at<float>(2,2) = 0.999804;
    // Tcws[0].at<float>(2,3) = 5.3258;

    // Tcws[1].at<float>(0,0) = 0.99991554;
    // Tcws[1].at<float>(0,1) = 0.00565967;
    // Tcws[1].at<float>(0,2) = -0.01169705;
    // Tcws[1].at<float>(0,3) = -0.503096;
    // Tcws[1].at<float>(1,0) = -0.00575021;
    // Tcws[1].at<float>(1,1) = 0.9999536;
    // Tcws[1].at<float>(1,2) = -0.00772146;
    // Tcws[1].at<float>(1,3) = 0.0575373;
    // Tcws[1].at<float>(2,0) = 0.01165281;
    // Tcws[1].at<float>(2,1) = 0.00778807;
    // Tcws[1].at<float>(2,2) = 0.9999018;
    // Tcws[1].at<float>(2,3) = 0.564036;

    // //map point: 2.92129, 0.894701, 8.56214
    // cv::Mat x3D = (cv::Mat_<float>(3,1) << 2.92129, 0.894701, 8.56214);
    

    // int nFeatures = 1000;
    // float fScaleFactor = 1.2;
    // int nLevels = 8;
    // int fIniThFAST = 5;
    // int fMinThFAST = 2;

    // cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    // K.at<float>(0,0) = 1406;
    // K.at<float>(1.1) = 1406;
    // K.at<float>(0,2) = 960;
    // K.at<float>(1,2) = 600;

    // cv::Mat distCoef(4,1,CV_32F);

    // ORBextractor* mpIniORBextractor = new ORBextractor(
    //     2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // cv::Mat im, imGray;
    // KeyFrame* pKF;
    // MapPoint* pMP;

    // vector<vector<cv::KeyPoint>> interestedPoints;
    // interestedPoints.resize(8);
    // interestedPoints[0].push_back(cv::KeyPoint(500, 500, 31, 0, 10));
    // for(int i = 0; i < 2; i++){
    //     im = cv::imread(argo_img_files[i]);
    //     cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
    //     Frame frame(imGray, i, mpIniORBextractor, mpVocabulary, K, distCoef, 0, 0, interestedPoints);
    //     frame.ComputeBoW();
    //     frame.SetPose(Tcws[i]);
    //     pKF = new KeyFrame(frame,mpMap,mpKeyFrameDatabase);
    //     if(i == 0){
    //         pMP = new MapPoint(x3D, pKF, mpMap);
    //         mpMap->AddMapPoint(pMP);
    //     }
    //     pMP->AddObservation(pKF,i);
    //     pKF->AddMapPoint(pMP,i);
    //     pMP->ComputeDistinctiveDescriptors();
    //     pMP->UpdateNormalAndDepth();

    //     pKF->UpdateConnections();
    //     mpMap->AddKeyFrame(pKF);
    //     mpKeyFrameDatabase->add(pKF);
    // }

    //Create Drawers. These are used by the Viewer
    // string strSettingsFile = "Examples/Monocular/TUM1.yaml";
    // FrameDrawer* mpFrameDrawer = new FrameDrawer(mpMap, true);
    // MapDrawer* mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);
    // drawGraph(mpMapDrawer, mpFrameDrawer);


    //point3D ID to Point3D
    unordered_map<int, Point3D> points3D;

    //image ID to Point2D vector
    unordered_map<int, vector<Point2D>> points2D;
    readPoints3D("experiments/data/points3D.txt", points3D);
    readPoints2D("experiments/data/points2D.txt", points2D);
    

    //暫時記錄imageid to keyframes, 之後用來更新MapPoint的attributes
    vector<KeyFrame*> addedKeyFrames;
    addedKeyFrames.resize(19);

    vector<vector<cv::KeyPoint>> interestedPoints;
    interestedPoints.resize(8);
    for(auto it : points3D){
        Point3D& point3D = it.second;

        //新增MapPoint, 新加的constructor
        //之後要補上 mnFirstKFid, mnFirstFrame, mpRefKF
        cv::Mat pos = (cv::Mat_<float>(3,1) << point3D.x, point3D.y, point3D.z);
        MapPoint* pMP = new MapPoint(pos, pMap);

        vector<int>::iterator pit = find(point3D.imageIds.begin(), point3D.imageIds.end(), 5);
        if(pit != point3D.imageIds.end()){
            auto idx = pit - point3D.imageIds.begin();
            int point2DId = point3D.points2dIds[idx];
            Point2D& point2D = points2D[5][point2DId];
            int level;
            float scale;
            discretizeScale(point2D.scale, 1.2, 8, level, scale);
            interestedPoints[level].push_back(
                cv::KeyPoint(point2D.x/scale, point2D.y/scale, 31*scale, point2D.orientation, 0, level));
        }
    }

    




    return 0;
}