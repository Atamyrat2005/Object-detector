# Real-Time Object Detector with OpenCV

A simple yet powerful real-time object detection application that uses a pre-trained SSD MobileNetV3 model with OpenCV to identify and label 80 common objects from a live webcam feed.

## 📸 Demo

![0_xfXdebLeaMXt3Vct](https://github.com/user-attachments/assets/05be7e62-3f44-4408-83f8-f3c3ebb66c2a)

A sample image showing object detection in action.

## ✨ Features

Real-Time Detection: Analyzes your webcam feed frame-by-frame.

80 Object Classes: Detects a wide range of common objects from the COCO dataset (e.g., person, car, dog, bottle, cell phone).

High Performance: Uses the fast and lightweight SSD (Single Shot Detector) with a MobileNetV3 backbone, suitable for running on most modern CPUs.

Clear Visualization: Draws bounding boxes around detected objects and labels them with the class name and confidence score.

## 🛠️ How It Works

This project leverages the Deep Neural Network (DNN) module available in OpenCV.

Model Loading: The script loads a pre-trained SSD MobileNetV3 model. This model was trained on the COCO (Common Objects in Context) dataset.

frozen_inference_graph.pb: The file containing the trained model weights.

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt: The model's configuration/architecture file.

Video Capture: It captures video from your default webcam using cv2.VideoCapture(0).

Frame Processing: In a continuous loop, each frame from the webcam is passed to the model.

Detection: The model predicts the classId, confidence score, and bounding box coordinates for all objects it finds in the frame.

Rendering: The script draws rectangles and text labels onto the frame for any object detected with a confidence score above 50% (threshold=0.5).

Display: The final processed frame is displayed in a window titled "Output".

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

Prerequisites

Python 3.6+

A webcam connected to your computer

Installation

Clone the repository:

Generated bash
git clone https://github.com/Atamyrat2005/Object-detector.git
cd Object-detector


Install the required Python libraries:
For convenience, I've listed the dependencies. You can install them using pip.

Generated bash
pip install opencv-python numpy
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Tip: It's good practice to create a requirements.txt file with opencv-python and numpy inside it and install with pip install -r requirements.txt)

Usage

Once the installation is complete, run the main script from your terminal:

Generated bash
python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

A window titled "Output" should appear, showing your webcam feed with objects being detected and labeled in real-time.

To stop the program, press the q key while the "Output" window is active, or close the window.

## 📂 Project Structure
Generated code
Object-detector/
├── coco.names                          # List of all 80 object names the model can detect
├── frozen_inference_graph.pb           # The pre-trained model weights
├── main.py                             # The main Python script to run the application
└── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt # The model configuration
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
## 💡 Possible Future Improvements

Add command-line arguments to allow processing a pre-recorded video file or a static image.

Allow the user to adjust the confidence threshold via a command-line argument.

Implement a feature to save the output video with the bounding boxes.

Test and add support for other pre-trained models.

## 📄 License

This project is open-source. Feel free to use and modify it. If you'd like to include a formal license, the MIT License is a great choice.
