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
#include<thread>
#include "BoostArchiver.h"

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

void readPoints2D(string filename, map<int, vector<Point2D>>& points2D){
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

void readPoses(string filename, unordered_map<int, cv::Mat>& imGrays, unordered_map<int, cv::Mat>& Tcws, unordered_map<int, int>& cameraIDs){
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
        cameraIDs[imageID] = cameraID;
    }
}

void readCameras(string filename, unordered_map<int, cv::Mat>& Ks, unordered_map<int, cv::Mat>& distCoefs){
    string line;
    ifstream camerasFile(filename);
    getline(camerasFile, line);
    getline(camerasFile, line);
    getline(camerasFile, line);
    int cameraID;
    int width, height;
    float f, cx, cy, k;
    string model;
    
    while(getline(camerasFile, line)){
        istringstream iss(line);
        iss >> cameraID >> model >> width >> height >> f >> cx >> cy >> k;
        
        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = f;
        K.at<float>(1,1) = f;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        cv::Mat distCoef(4,1,CV_32F);
        Ks[cameraID] = K;
        distCoefs[cameraID] = distCoef;
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

void constructKeyFrames(
                    map<int, vector<Point2D>>& points2D,
                    unordered_map<int, int>& cameraIDs,
                    unordered_map<int, cv::Mat>& imGrays,
                    unordered_map<int, cv::Mat>& Tcws,
                    ORBextractor* pORBextractor,
                    ORBVocabulary* pVocabulary,
                    unordered_map<int, cv::Mat>& Ks,
                    unordered_map<int, cv::Mat>& distCoefs,
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
        
        int cameraID = cameraIDs[imID];
        Frame frame(imGrays[imID], 0, pORBextractor, pVocabulary, Ks[cameraID], distCoefs[cameraID], 0, 0, interestedPoints);
        // Frame frame(imGrays[imID], 0, pORBextractor, pVocabulary, Ks[cameraID], distCoefs[cameraID], 0, 0);
        frame.ComputeBoW();
        frame.SetPose(Tcws[imID]);
        pKeyFrames[imID] = new KeyFrame(frame,pMap,pKeyFrameDatabase);
        pKeyFrames[imID]->mbFixed = true;
        cout << imID << " " << pKeyFrames[imID]->mnId << endl;
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

void SaveMap(const string &filename, Map* pMap, KeyFrameDatabase* pKeyFrameDatabase)
{
    unique_lock<mutex>MapPointGlobal(MapPoint::mGlobalMutex);
    std::ofstream out(filename, std::ios_base::binary);
    if (!out)
    {
        cerr << "Cannot Write to Mapfile: " << filename << std::endl;
        exit(-1);
    }
    cout << "Saving Mapfile: " << filename << std::flush;
    boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    oa << pMap;
    oa << pKeyFrameDatabase;
    cout << " ...done" << std::endl;
    out.close();
}

bool LoadMap(const string &filename, Map*& pMap, KeyFrameDatabase*& pKeyFrameDatabase, ORBVocabulary* pVocabulary)
{
    unique_lock<mutex>MapPointGlobal(MapPoint::mGlobalMutex);
    std::ifstream in(filename, std::ios_base::binary);
    if (!in)
    {
        cerr << "Cannot Open Mapfile: " << filename << " , You need create it first!" << std::endl;
        return false;
    }
    cout << "Loading Mapfile: " << filename << std::flush;
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> pMap;
    ia >> pKeyFrameDatabase;
    pKeyFrameDatabase->SetORBvocabulary(pVocabulary);
    cout << " ...done" << std::endl;
    cout << "Map Reconstructing" << flush;
    vector<ORB_SLAM2::KeyFrame*> vpKFS = pMap->GetAllKeyFrames();
    unsigned long mnFrameId = 0;
    for (auto it:vpKFS) {
        it->SetORBvocabulary(pVocabulary);
        it->ComputeBoW();
        if (it->mnFrameId > mnFrameId)
            mnFrameId = it->mnFrameId;
    }
    Frame::nNextId = mnFrameId;
    cout << " ...done" << endl;
    in.close();
    return true;
}

int main(int argc, char **argv)
{
    ORBVocabulary* pVocabulary = new ORBVocabulary();
    cout << "Load Vocabulary File" << endl;
    const string strVocFile("Vocabulary/ORBvoc.bin");
    pVocabulary->loadFromBinaryFile(strVocFile);

    cout << "Create KeyFrame database and Map" << endl;
    KeyFrameDatabase* pKeyFrameDatabase = new KeyFrameDatabase(pVocabulary);
    Map* pMap = new Map();

    // int nFeatures = 3000;
    // float fScaleFactor = 1.4;
    // int nLevels = 10;
    // int fIniThFAST = 5;
    // int fMinThFAST = 2;

    // ORBextractor* pORBextractor = new ORBextractor(
    //     nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // //point3D ID to Point3D
    // unordered_map<int, Point3D> points3D;
    // //image ID to Point2D vector
    // map<int, vector<Point2D>> points2D;
    // unordered_map<int, int> cameraIDs;
    // unordered_map<int, KeyFrame*> pKeyFrames;
    // unordered_map<int, cv::Mat> imGrays;
    // unordered_map<int, cv::Mat> Tcws;
    // unordered_map<int, cv::Mat> Ks;
    // unordered_map<int, cv::Mat> distCoefs;
    // unordered_map<int, vector<int>> orders;
    // unordered_map<int, vector<int>> accCounts;

    // cout << "read points3D" << endl;
    // readPoints3D("experiments/data2/points3D.txt", points3D);
    // cout << "read points2D" << endl;
    // readPoints2D("experiments/data2/points2D.txt", points2D);
    // cout << "read poses" << endl;
    // readPoses("experiments/data2/poses.txt", imGrays, Tcws, cameraIDs);
    // cout << "read cameras" << endl;
    // readCameras("experiments/data2/cameras.txt", Ks, distCoefs);

    // cout << "construct KeyFrames" << endl;
    // constructKeyFrames(
    //     points2D, cameraIDs, imGrays, Tcws, pORBextractor,
    //     pVocabulary, Ks, distCoefs, fScaleFactor,
    //     nLevels, pKeyFrameDatabase, pMap,
    //     pKeyFrames, orders, accCounts);

    // // //add MapPoints to Map
    // cout << "Add MapPoints" << endl;
    // for(auto it : points3D){
    //     Point3D& point3D = it.second;

    //     //新增MapPoint, 隨意拿第一個ImageID做為first keyframe
    //     cv::Mat pos = (cv::Mat_<float>(3,1) << point3D.x, point3D.y, point3D.z);
    //     MapPoint* pMP = new MapPoint(pos, pKeyFrames[point3D.imageIds[0]],pMap);
    //     pMP->mbFixed = true;
    //     for(int i = 0; i < point3D.imageIds.size(); i++){
    //         int imID = point3D.imageIds[i];
    //         int point2DId = point3D.points2dIds[i];
    //         int keypointId, offset;
    //         for(int level = 0; level < nLevels; level++){
    //             if(orders[imID][point2DId] < accCounts[imID][level]){
    //                 offset = accCounts[imID][level] - orders[imID][point2DId];
    //                 keypointId = pKeyFrames[imID]->mvAccKeyPoints[level] - offset;
    //                 break;
    //             }
    //         }
    //         pKeyFrames[imID]->AddMapPoint(pMP, keypointId);
    //         pMP->AddObservation(pKeyFrames[imID], keypointId);
    //         pMP->ComputeDistinctiveDescriptors();
    //         pMP->UpdateNormalAndDepth();
    //         pMap->AddMapPoint(pMP);
    //     }
    // }

    // //Initialize the Local Mapping thread and launch
    // LocalMapping* pLocalMapper = new LocalMapping(pMap, 1);
    // thread* ptLocalMapping = new thread(&LocalMapping::Run,pLocalMapper);

    // //Initialize the Loop Closing thread and launch
    // LoopClosing* pLoopCloser = new LoopClosing(pMap, pKeyFrameDatabase, pVocabulary, 0);
    // thread* ptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, pLoopCloser);

    // pLocalMapper->SetLoopCloser(pLoopCloser);
    // pLoopCloser->SetLocalMapper(pLocalMapper);

    // chrono::milliseconds timespan(500);
    
    // //Add KeyFrame to Map
    // cout << "Add KeyFrames" << endl;
    // for(auto it: pKeyFrames){
    //     cout << "insert KeyFrame " << it.first << endl;
    //     it.second->UpdateConnections();
    //     pKeyFrameDatabase->add(it.second);
    //     pMap->AddKeyFrame(it.second);
    //     pLocalMapper->InsertKeyFrame(it.second);
    //     this_thread::sleep_for(timespan);
    // }

    // cout << "Enter to continue...";
    // getchar();

    // pLocalMapper->RequestFinish();
    // pLoopCloser->RequestFinish();

    // // Wait until all thread have effectively stopped
    // while(!pLocalMapper->isFinished() || !pLoopCloser->isFinished() || pLoopCloser->isRunningGBA())
    // {
    //     std::this_thread::sleep_for(std::chrono::microseconds(5000));
    // }

    // cout << pMap->GetAllMapPoints().size() << " MapPoints" << endl;

    // SaveMap("experiments/data2/ColmapMap.bin", pMap, pKeyFrameDatabase);


    Map* pNewMap;
    KeyFrameDatabase* pNewKeyFrameDatabase;
    LoadMap("experiments/data2/ColmapMap.bin", pNewMap, pNewKeyFrameDatabase, pVocabulary);

    // // //Create Drawers. These are used by the Viewer
    string strSettingsFile = "Examples/Argoverse/argoverse.yaml";
    FrameDrawer* mpFrameDrawer = new FrameDrawer(pNewMap, true);
    MapDrawer* mpMapDrawer = new MapDrawer(pNewMap, strSettingsFile);
    drawGraph(mpMapDrawer, mpFrameDrawer);


    return 0;
}