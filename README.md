This is a project to caliberate a camera and then use the obtained parameters to detect dimensions of a planer object.

You can use the (code to click images of chessboard pattern.ipynb) to click multiple images from different angles and store them in a repository.
These images can now be used by the Module 1 Python code.ipynb file to fing the camera parameters

This repsitory also contains a web app to diplay dimensions of an object in real time with video capturing based on flask and HTML.


The IPYNB file contains full python code.

Steps to Run the Web APP

1. The index.HTML files need to be in a directory named as templates and the app.py file should be in the same directory as template and other numpy files.
2. The camera should be connected to the device.
3. The app.py file should be exicuted to make a local host server (while executing app.py please check that the OpenCV VideoCapture function is using the desired camera to capture the video feed)
4. Click on the local host server, this will direct you to the front-end of the application.
5. Click on the button "Start Measurement" and place your object in the view of the camera. ( Ensure that the object is at a fixed distance{you can update this in the app.py file} from the camera and the object plane is parallel to the camera plane.)
6. The dimensions of the object would be visible below the real-time camera feed.
