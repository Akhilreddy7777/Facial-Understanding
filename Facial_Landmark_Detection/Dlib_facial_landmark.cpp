

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#define DLIB_JPEG_SUPPORT
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



using namespace cv;


using namespace dlib;
using namespace std;
ofstream myfile;

// ----------------------------------------------------------------------------------------

int main(int argc, const char** argv)
{
	String ModelPath = "./shape_predictor_68_face_landmarks.dat";
	std::vector<String> filesnames;
	String delimiter = ".jpg";
	String folder = "./faces/";
	glob(folder, filesnames);

	try
	{
		// This example takes in a shape model file and then a list of images to
		// process.  We will take these filenames in as command line arguments.
		// Dlib comes with example images in the examples/faces folder so give
		// those as arguments to this program.
		
		

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		frontal_face_detector detector = get_frontal_face_detector();
		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		shape_predictor sp;
		deserialize(ModelPath) >> sp;


		image_window win, win_faces;

		//edited
		//myfile.open("./facial_features.csv");
		// Loop over all the images provided on the command line.
		for (size_t i = 0; i < filesnames.size(); i++)
		{
			cout << "processing image " << filesnames[i] << endl;
			array2d<rgb_pixel> img;
			load_image(img, filesnames[i]);
			// Make the image larger so we can detect small faces.
			//pyramid_up(img);

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			std::vector<dlib::rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img, dets[j]); 
				//cout << "number of parts: " << shape.num_parts() << endl;
				//cout << "pixel position of first part:  " << shape.part(0) << endl;
				//cout << "pixel position of second part: " << shape.part(1) << endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
				String token = filesnames[i].substr(0, filesnames[i].find(delimiter));
				myfile.open(token+".csv");
				for (int ct = 0; ct < 68; ct++)
				{
					myfile << filesnames[i] << "," << "faces_detected" << "," << dets.size() << "," << shape.part(ct) << std::endl;
				}
				myfile.close();

			}

			// Now let's view our face poses on the screen.
			/*win.clear_overlay();
			win.set_image(img);
			win.add_overlay(render_face_detections(shapes));

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			dlib::array<array2d<rgb_pixel> > face_chips;
			extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));

			cout << "Hit enter to process the next image..." << endl;
			cin.get();
			*/
		}
		//myfile.close();
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
	
}

// ----------------------------------------------------------------------------------------

