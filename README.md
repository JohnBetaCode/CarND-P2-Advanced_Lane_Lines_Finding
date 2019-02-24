## P2 - Advanced Road Lane Lines Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

---
**Descripton:**

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle (depending on lines curvature). Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm then extract some of their features. This project detect lane lines and features like curvature, finding a surface projection and a relation between images world and real world. I used the tools that I learned about in the lesson (Computer Vision Fundamentals, Camera Calibration, Gradients and Color Spaces, and advanced computer vision from Udacity's Self driving car NanoDegree).

---
**Goals**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
**Contents description**

The images for camera calibration are stored in the folder called `camera_cal`. The images in `test_images` are for testing the pipeline on single frames.

Outputs from each stage of the pipeline were saved in the folder called `output_images`, and include a description in the writeup file `CarND-P2-Advanced_Lane_Lines_Finding.md` for the project of what each image shows. The video called `project_video.mp4` is the video which pipeline should work well on. 

The `challenge_video.mp4` video is an extra (and optional) challenge to test the pipeline under somewhat trickier conditions. The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious, I encourage you to go out and take video of your own, calibrate your camera and show how you would implement this project from scratch!

---

### **How to run**

To run the pipeline for images and videos just run in a prompt the command:

```clear && CarND-P2-Advanced_Lane_Lines_Finding.py```

Tested on: python 2.7 (3.X should work), OpenCV 3.0.0 (Higher version should work), UBUNTU 16.04.

Feel free to change any input argument of any function explained in `CarND-P2-Advanced_Lane_Lines_Finding.md`.

---
