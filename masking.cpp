
// Set the desired face dimensions. 
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Try to set the camera resolution.
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;


const char *windowName = "Masking";   // shown in the GUI window.
int BORDER = 8;  // Border between GUI elements to the edge of the image.


#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      // Escape character (27)
#endif


// Running mode for the Webcam-based interactive GUI program.
enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL,   MODE_END};
const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

int face_maks=1;

// Position of GUI buttons:
Rect m_rcBtnAdd1;
Rect m_rcBtnAdd2;
Rect m_rcBtnAdd3;
Rect m_rcBtnAdd4;
Rect m_rcBtnAdd5;


int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

double min_face_size=20;
double max_face_size=200;

// Get access to the webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    try {  
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}


Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    if (coord.y >= 0) {
        coord.y += textSize.height;
    }
    else {
        coord.y += img.rows - baseline + 1;
    }
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }

    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    return boundingRect;
}

// Draw a GUI button into the image, using drawString().
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Get the bounding box around the text.
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // Draw a filled rectangle around the text.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);

    // Draw the actual text that will be displayed, using anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(10,55,20));

    return rcButton;
}

bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

// Mouse event handler.
void onMouse(int event, int x, int y, int, void*)
{
    // We only care about left-mouse clicks, not right-mouse clicks or mouse movement.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    Point pt = Point(x,y);
    if (isPointInRect(pt, m_rcBtnAdd1)) {
        face_maks=1;
    }
    else if (isPointInRect(pt, m_rcBtnAdd2)) {
        face_maks=2;
    }

    else if (isPointInRect(pt, m_rcBtnAdd3)) {
        face_maks=3;
    }
	else if (isPointInRect(pt, m_rcBtnAdd4)) {
        //system("gedit");
	face_maks=4;
    }
	else if (isPointInRect(pt, m_rcBtnAdd5)) {
        //system("gedit");
	face_maks=5;
    }
    else {
        cout << "User clicked on the image" << endl;
       
        }
       
}
void addButton(Mat &displayedFrame)
{
	m_rcBtnAdd1 = drawButton(displayedFrame, "mask1" , Point(BORDER, BORDER));
        m_rcBtnAdd2 = drawButton(displayedFrame, "mask2", Point(m_rcBtnAdd1.x, m_rcBtnAdd1.y + m_rcBtnAdd1.height), m_rcBtnAdd1.width);
        m_rcBtnAdd3 = drawButton(displayedFrame, "mask3", Point(m_rcBtnAdd2.x, m_rcBtnAdd2.y + m_rcBtnAdd2.height), m_rcBtnAdd1.width);
        m_rcBtnAdd4 = drawButton(displayedFrame, "mask4", Point(m_rcBtnAdd3.x, m_rcBtnAdd3.y + m_rcBtnAdd3.height), m_rcBtnAdd1.width);
	m_rcBtnAdd5 = drawButton(displayedFrame, "mask5", Point(m_rcBtnAdd4.x, m_rcBtnAdd4.y + m_rcBtnAdd4.height), m_rcBtnAdd1.width);


}

Mat putMask(Mat src,Mat mask,Point center,Size face_size)
{
    Mat mask1,src1;
    resize(mask,mask1,face_size);   
     Rect roi(center.x - face_size.width/2, center.y - face_size.width/2, face_size.width, face_size.width);
    src(roi).copyTo(src1);
 	
    // to make the white region transparent
    Mat mask2,m,m1;
    cvtColor(mask1,mask2,CV_BGR2GRAY);
    threshold(mask2,mask2,230,255,CV_THRESH_BINARY_INV);
 
    vector<Mat> maskChannels(3),result_mask(3);
    split(mask1, maskChannels);
    bitwise_and(maskChannels[0],mask2,result_mask[0]);
    bitwise_and(maskChannels[1],mask2,result_mask[1]);
    bitwise_and(maskChannels[2],mask2,result_mask[2]);
    merge(result_mask,m );         //    imshow("m",m);
 
    mask2 = 255 - mask2;
    vector<Mat> srcChannels(3);
    split(src1, srcChannels);
    bitwise_and(srcChannels[0],mask2,result_mask[0]);
    bitwise_and(srcChannels[1],mask2,result_mask[1]);
    bitwise_and(srcChannels[2],mask2,result_mask[2]);
    merge(result_mask,m1 );        //    imshow("m1",m1);
 
    addWeighted(m,1,m1,1,0,m1);    //    imshow("m2",m1);
     
    m1.copyTo(src(roi));
 
    return src;
}



Mat detectFace(Mat &image,Mat mask)
{
    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade( "data/haarcascade_frontalface_alt.xml" );
 
    // Detect faces
    std::vector<Rect> faces;
 
    face_cascade.detectMultiScale( image, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
     
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {   // Lets only track the first face, i.e. face[0]
        min_face_size = faces[0].width*0.7;
        max_face_size = faces[0].width*1.5;
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );    
	image=putMask(image,mask,center,Size( faces[i].width, faces[i].height));
    }   
	 return image;
}
 

// Main loop that runs forever, until the user hits Escape to quit.
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture)
{
	Mat fmask;
	Mat mask3 = imread("images/6.jpg"); 
	Mat mask2 = imread("images/2.jpg"); 
	Mat mask1 = imread("images/1.jpg"); 
	Mat mask4 = imread("images/11.jpg"); 
	Mat mask5 = imread("images/9.jpg"); 
	
	   
	// Run forever, until the user hits Escape to "break" out of this loop.
    while (true) {

        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }

        // Get a copy of the camera frame that we can draw onto.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);
	 char keypress = waitKey(20);


	addButton(displayedFrame);
	switch(face_maks)
	{
		case 1: fmask=mask1; break;
		case 2: fmask=mask2; break;
		case 3: fmask=mask3; break;
		case 4: fmask=mask4; break;
		case 5: fmask=mask5; break;
	}
	detectFace(displayedFrame,fmask);

	imshow(windowName, displayedFrame);
        if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }

    }//end while
}


int main(int argc, char *argv[])
{
    VideoCapture videoCapture;

    system("espeak Hi");
    // Load the face and 1 or 2 eye detection XML classifiers.
    //initDetectors(faceCascade, eyeCascade1, eyeCascade2);

    cout << endl;
    cout << "Hit 'Escape' in the GUI window to quit." << endl;

    // Allow the user to specify a camera number, 
    int cameraNumber = 0;  
    if (argc > 1) {
        cameraNumber = atoi(argv[1]);
    }

    // Get access to the webcam.
    initWebcam(videoCapture, cameraNumber);

    // Try to set the camera resolution.
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Create a GUI window for display on the screen.
    namedWindow(windowName,0); 
    // Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
    setMouseCallback(windowName, onMouse, 0);

	
    // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
    recognizeAndTrainUsingWebcam(videoCapture);


    return 0;
}
