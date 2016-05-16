

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>



using namespace cv;
using namespace cv::face;
using namespace std;

const char TRAINING_MODE = '1';
const char DETECTING_MODE = '0';

string face_cascade_name = "../data/haarcascade_frontalface_alt.xml";    //Nazwa kaskady któr¹ wykorzystujemy do rozpoznania twarzy 
CascadeClassifier face_cascade;

void detectFace(Mat &img, int im_width, int im_height)
{
	vector<Rect> faces;                            //Utworzenie bufora na twarze 
	Mat img_gray;                                //Utworzenie bufora na obrazek w odcieniach szarosci 
	Mat img_cut;
	Mat img_resized;
	Mat resized_gray;
	//vector<int> compression_params;
	//compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	//compression_params.push_back(100);
	cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);                //Konwersja obrazu do odcieni szarosci 
	face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50));
	for (unsigned i = 0; i < faces.size(); i++)
	{
		img_cut = img(faces[i]);
		resize(img_cut, img_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		cvtColor(img_resized, resized_gray, cv::COLOR_BGR2GRAY);
	}                    //Pokazanie obrazka w onkie o nazwie "Hello Face !" 
						 //putText(resized_gray, file_name, Point(50, 100), FONT_HERSHEY_SIMPLEX, 0.5, 2, 0 ); 
						 //imwrite(file_name.c_str(), resized_gray, compression_params);
	img = resized_gray;
	if (img.empty())
		cout << "cos zle w detectFace";
	//resized_gray.convertTo(img, 3);
}

void read_csv(const string & filename, vector<Mat> & images, vector<int> & labels)
{
	Mat temp;
	ifstream file(filename.c_str(), ifstream::in);
	if (!file)
	{
		cout << "Otwarcie pliku " << filename << " nie powiodlo sie.";
		return;
	}
	string line, path, classlabel;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, ';');
		getline(liness, classlabel);
		classlabel.erase(0, 1);
		path.erase(path.length() - 1, path.length());
		if (!path.empty() && !classlabel.empty())
		{
			temp = imread(path, 1);
			cout << "processing: " << path << endl;
			detectFace(temp, 92, 112);
			if (temp.empty())
				cout << "wrong " + path + " " << endl;
			else {
				images.push_back(temp);
				labels.push_back(atoi(classlabel.c_str()));

				//while (waitKey(0) != 27) { //to jest Esc, jakby co
					//imshow("Face in the loop", temp);
				//}
			}
		}
	}
}

//0 ../data/trainedModel.xml ../data/finalny_test.jpg
//1 D:/faces.csv ../data/trainedModel.xml
//format argumentów = tryb pracy, sciezka do csv/wytrenmodelu, gdzie zapisac model/plik z samplem
//plik csv wygenerowac skryptem pythonowym albo recznie, w zaleznosci od tego ile masz czasu xD

int main(int argc, char *argv[])
{
	vector<Mat> images;
	vector<int> labels;
	string test_name = "../data/test_3.jpg";
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	int plabel;
	double distance;

	if (!face_cascade.load(face_cascade_name))        //£adowanie pliku ze sprawdzeniem poprawnoci 
	{
		cout << "Nie znaleziono pliku " << face_cascade_name << ".";
		return -2;
	}

	if (argc < 4) {
		cout << "Zbyt ma³o argumentów (" << argc << ") podaj argumenty w formacie tryb i sciezka do pliku";
		return -2;
	}

	if (*argv[1] == TRAINING_MODE) {
		cout << "TRAINING MODE";
		read_csv(string(argv[2]), images, labels);
		cout << "Odczytywanie skonczone";
		model->train(images, labels);
		try {
			model->save("../data/trainedModel.xml");
			cout << "Pomyslnie nauczono i zapisano model.";
		}
		catch (cv::Exception& e) {
			cerr << "Bledna sciezka do pliku z modelem" << endl;
			exit(1);
		}

	}

	else if (*argv[1] == DETECTING_MODE) {
		cout << "DETECTING MODE";
		string trainedModelPath = string(argv[2]);
		ifstream file(trainedModelPath.c_str(), ifstream::in);
		if (!file)
		{
			cout << "Otwarcie pliku " << trainedModelPath << " nie powiodlo sie.";
			return -2;
		}
		model->load(trainedModelPath);
		//namedWindow("sample", CV_WINDOW_AUTOSIZE);
		//namedWindow("predicted", CV_WINDOW_AUTOSIZE);
		Mat recFace = imread(string(argv[3]), 1);
		if (!recFace.data)
		{
			cout << "Nie odnaleziono " << test_name << ".";
			return -2;
		}

		detectFace(recFace, 92, 112);
		model->predict(recFace, plabel, distance);
		cout << plabel << " distance: " << distance;

		while (waitKey(0) != 27) { //to jest Esc, jakby co
			imshow("Face to be Recognized", recFace);
		}
	}
	else {
		cout << "bledne argumenty " << argv[1];
		return -2;
	}

	//cout << plabel << " distance: " << distance;
	//int i = 0;
	//imshow("predicted", images[5 * plabel + (i % 5)]);
	//model->save("../data/trainedModel.xml");
	//return 0;
}