#include<string>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<opencv2/core/core.hpp>
#include<numeric>      // std::iota
#include<System.h>
#include<Eigen/Dense>
#include"Converter.h"

using namespace std;
using namespace ORB_SLAM2;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


struct Point3D{
    float x,y,z;
    vector<int> imageIds;
    vector<int> points2dIds;
};

struct Point2D{
    float x,y,scale,orientation;
    bool operator < (const Point2D& other) const{
        return scale < other.scale;
    }
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

void readPoses(string filename, unordered_map<int, cv::Mat>& imGrays, unordered_map<int, cv::Mat>& Tcws){
    string line;
    ifstream posesFile(filename);
    getline(posesFile, line);
    int imageID, cameraID;
    float qw,qx,qy,qz,tx,ty,tz;
    string imageFilename;
    Converter converter;
    
    while(getline(posesFile, line)){
        istringstream iss(line);
        iss >> imageID >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraID >> imageFilename;
        Eigen::Quaternion<double> q(qw, qx, qy, qz);
        Eigen::Matrix<double,3,1> v;
        v << tx, ty, tz;
        cv::Mat Tcw = converter.toCvSE3(q.toRotationMatrix(), v);
        cv::Mat im, imGray;
        im = cv::imread(imageFilename);
        cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
        imGrays[imageID] = imGray;
        Tcws[imageID] = Tcw;
    }
}

void discretizeScale(float scaleColmap, float scaleFactor, int maxLevel, int& level, float& scale){
    level = 0;
    scale = 1;
    while(scaleColmap > scaleFactor && level < maxLevel-1){
        level++;
        scaleColmap /= scaleFactor;
        scale *= scaleFactor;
    }
}

void constructKeyFrames(unordered_map<int, vector<Point2D>>& points2D, 
                     unordered_map<int, cv::Mat>& imGrays,
                     unordered_map<int, cv::Mat>& Tcws,
                     ORBextractor* pORBextractor,
                     ORBVocabulary* pVocabulary,
                     cv::Mat& K,
                     cv::Mat& distCoef,
                     float fScaleFactor,
                     int nLevels,
                     KeyFrameDatabase* pKeyFrameDatabase,
                     Map* pMap,
                     unordered_map<int, KeyFrame*>& pKeyFrames,
                     unordered_map<int, vector<int>>& orders,
                     unordered_map<int, vector<int>>& accCounts
                     ){
    for(auto it: points2D){
        int imID = it.first;
        vector<Point2D>& colmapPoints2D = it.second;
        vector<vector<cv::KeyPoint>> interestedPoints;
        vector<size_t> sorted_index = sort_indexes(colmapPoints2D);
        vector<int>& accCount = accCounts[imID];
        accCount.resize(nLevels);
        interestedPoints.resize(nLevels);

        //points2D id to its order of scale
        vector<int>& order = orders[imID];
        order.resize(colmapPoints2D.size());
        for(int i = 0; i < order.size(); i++){
            order[sorted_index[i]] = i;
        }

        for(int i = 0, j = 0; i < (int)colmapPoints2D.size(); i++){
            const Point2D& point2D = colmapPoints2D[sorted_index[i]];
            int level;
            float scale;
            discretizeScale(point2D.scale, fScaleFactor, nLevels, level, scale);
            interestedPoints[level].push_back(
                cv::KeyPoint(point2D.x/scale, point2D.y/scale, 31*scale, point2D.orientation, 0, level));
            while(level > j){
                accCount[j++] = i;
            }
        }
        accCount[nLevels-1] = colmapPoints2D.size();
        Frame frame(imGrays[imID], 0, pORBextractor, pVocabulary, K, distCoef, 0, 0, interestedPoints);
        frame.ComputeBoW();
        frame.SetPose(Tcws[imID]);
        pKeyFrames[imID] = new KeyFrame(frame,pMap,pKeyFrameDatabase);
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
    ORBVocabulary* mpVocabulary = new ORBVocabulary();
    cout << "Load Vocabulary File" << endl;
    const string strVocFile("Vocabulary/ORBvoc.txt");
    mpVocabulary->loadFromTextFile(strVocFile);

    cout << "Create KeyFrame database and Map" << endl;
    KeyFrameDatabase* mpKeyFrameDatabase = new KeyFrameDatabase(mpVocabulary);
    Map* pMap = new Map();

    int nFeatures = 1000;
    float fScaleFactor = 1.2;
    int nLevels = 8;
    int fIniThFAST = 5;
    int fMinThFAST = 2;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = 1406;
    K.at<float>(1.1) = 1406;
    K.at<float>(0,2) = 960;
    K.at<float>(1,2) = 600;

    cv::Mat distCoef(4,1,CV_32F);

    ORBextractor* mpIniORBextractor = new ORBextractor(
        2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //point3D ID to Point3D
    unordered_map<int, Point3D> points3D;
    //image ID to Point2D vector
    unordered_map<int, vector<Point2D>> points2D;

    unordered_map<int, KeyFrame*> pKeyFrames;
    unordered_map<int, cv::Mat> imGrays;
    unordered_map<int, cv::Mat> Tcws;
    unordered_map<int, vector<int>> orders;
    unordered_map<int, vector<int>> accCounts;

    readPoints3D("experiments/data/points3D.txt", points3D);
    readPoints2D("experiments/data/points2D.txt", points2D);
    readPoses("experiments/data/poses.txt", imGrays, Tcws);

    constructKeyFrames(
        points2D, imGrays, Tcws, mpIniORBextractor,
        mpVocabulary, K, distCoef, fScaleFactor,
        nLevels, mpKeyFrameDatabase, pMap,
        pKeyFrames, orders, accCounts);


    for(auto it : points3D){
        Point3D& point3D = it.second;

        //新增MapPoint, 隨意拿第一個ImageID做為first keyframe
        cv::Mat pos = (cv::Mat_<float>(3,1) << point3D.x, point3D.y, point3D.z);
        MapPoint* pMP = new MapPoint(pos, pKeyFrames[point3D.imageIds[0]],pMap);
        for(int i = 0; i < point3D.imageIds.size(); i++){
            int imID = point3D.imageIds[i];
            int point2DId = point3D.points2dIds[i];
            int keypointId, offset;
            for(int level = 0; level < nLevels; level++){
                if(orders[imID][point2DId] < accCounts[imID][level]){
                    offset = accCounts[imID][level] - orders[imID][point2DId];
                    keypointId = pKeyFrames[imID]->mvAccKeyPoints[level] - offset;
                    break;
                }
            }
            pKeyFrames[imID]->AddMapPoint(pMP, keypointId);
            pMP->AddObservation(pKeyFrames[imID], keypointId);
            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
            pMap->AddMapPoint(pMP);
        }
    }

    for(auto it: pKeyFrames){
        it.second->UpdateConnections();
        mpKeyFrameDatabase->add(it.second);
        pMap->AddKeyFrame(it.second);
    }

     //Create Drawers. These are used by the Viewer
    string strSettingsFile = "Examples/Monocular/TUM1.yaml";
    FrameDrawer* mpFrameDrawer = new FrameDrawer(pMap, true);
    MapDrawer* mpMapDrawer = new MapDrawer(pMap, strSettingsFile);
    drawGraph(mpMapDrawer, mpFrameDrawer);

    return 0;
}