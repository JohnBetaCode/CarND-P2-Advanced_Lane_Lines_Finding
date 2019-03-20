# =============================================================================
"""
Code Information:
    Date: 03/15/2019
	Programmer: John A. Betancourt G.
	Phone: +57 (311) 813 7206 / +57 (350) 283 51 22
	Mail: john.betancourt93@gmail.com / john@kiwicampus.com
    Web: www.linkedin.com/in/jhon-alberto-betancourt-gonzalez-345557129

Description: Project 2 - Udacity - self driving cars Nanodegree
    Advanced Road Lane Lines Finding

Tested on: 
    python 2.7 (3.X should work)
    OpenCV 3.0.0 (3.X or 4.X should work)
    UBUNTU 16.04
"""

# =============================================================================
# LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPEN
# =============================================================================
#importing useful packages
from __future__ import print_function, division
import numpy as np
import math
import cv2
import os

from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

warning_icon = cv2.imread("./writeup_files/icon_4.png", cv2.IMREAD_UNCHANGED)
critica_icon = cv2.imread("./writeup_files/icon_3.png", cv2.IMREAD_UNCHANGED)

# =============================================================================
# LANE LINES CLASS - LANE LINES CLASS - LANE LINES CLASS - LANE LINES CLASS - L
# =============================================================================
# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self, label):

        self.label = label

        # was the line detected in the last iteration?
        self.detected = False  

        #polynomial coefficients averaged over the last n iterations
        self.current_fit = None

        #History of polynomial coefficients
        self.hist_fit = []

        #radius of curvature of the line in some units
        self.radius_of_curvature = None 

        #x values for detected line pixels
        self.allx = None  

        #y values for detected line pixels
        self.ally = None  

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

        # Confidence of current polinomial fit
        self.fit_confidence = 0

        # ---------------------------------------
        # x values of the last n fits of the line
        self.recent_xfitted = [] 

        #average x values of the fitted line over the last n iterations
        self.bestx = None     

    def assing_fit(self, poly_fit, x_coords, y_coords, y_eval = 0, num_samples = 10):
      
        # detected line pixels assigment
        self.allx = x_coords # x values for detected line pixels
        self.ally = y_coords # y values for detected line pixels

        # Change state of lane line if there's poly fit
        self.detected = False if poly_fit is None else True

        # If there's more than 'num_samples' associations delete the first in history
        if ((len(self.hist_fit) and poly_fit is None) or 
            len(self.hist_fit) >= num_samples): 
            self.hist_fit.pop(0)

        # difference in fit coefficients between last and new fits
        if len(self.hist_fit) and poly_fit is not None: 
            self.diffs = self.hist_fit[-1] - poly_fit

        # Add new polynomial fit values 
        self.hist_fit.append(poly_fit)

        # Calculate new poly fit
        A = np.mean([arg_poly_fit[0] for arg_poly_fit in self.hist_fit])
        B = np.mean([arg_poly_fit[1] for arg_poly_fit in self.hist_fit])
        C = np.mean([arg_poly_fit[2] for arg_poly_fit in self.hist_fit])
        self.current_fit = [A, B, C]

        # if self.current_fit is not None:  
        #     self.fit_confidence = cross_entropy(self.current_fit, poly_fit)

        # if self.fit_confidence < 1.:

        # print(self, len(poly_fit), len(x_coords), len(y_coords))
 

    def __str__(self):

        str2print = "|label: {}\t|".format(self.label) \
            + "confi: {}\t|".format(round(self.fit_confidence), 3) \
            + "hist_fit: {}\t|".format(len(self.hist_fit)) 

        return str2print

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

# =============================================================================
# FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCT
# =============================================================================
# -----------------------------------------------------------------------------
# UTILS FUNCTIONS - UTILS FUNCTIONS - UTILS FUNCTIONS - UTILS FUNCTIONS - UTILS
def nothing(x): pass

def print_list_text(img_src, str_list, origin=(0, 0), color=(0, 255, 255), 
    thickness=2, fontScale=0.45,  y_space=20):

    """  prints text list in cool way
    Args:
        img_src: `cv2.math` input image to draw text
        str_list: `list` list with text for each row
        origin: `tuple` (X, Y) coordinates to start drawings text vertically
        color: `tuple` (R, G, B) color values of text to print
        thickness: `int` thickness of text to print
        fontScale: `float` font scale of text to print
        y_space: `int` [pix] vertical space between lines
    Returns:
        img_src: `cv2.math` input image with text drawn
    """

    for idx, strprint in enumerate(str_list):
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = (0, 0, 0), 
                    thickness = thickness+3, 
                    lineType = cv2.LINE_AA)
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = color, 
                    thickness = thickness, 
                    lineType = cv2.LINE_AA)

    return img_src

def line_intersection(line1, line2):

    """  Finds the intersection coordinate between two lines
    Args:
        line1: `tuple` line 1 to calculate intersection coordinate (X, Y) [pix]
        line2: `tuple` line 2 to calculate intersection coordinate (X, Y) [pix]
    Returns:
        inter_coord: `tuple` intersection coordinate between line 1 and line 2
    """

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

     # if lines do not intersect
    if div == 0:
       return 0, 0
       
    # Calculates intersection cord
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    inter_coord = [int(round(x)), int(round(y))]

    # Return X and Y cords of intersection
    return inter_coord

def dotline(src, p1, p2, color, thickness, Dl):

    """  draws a doted line on input image
    Args:
        src: `cv2.mat` source image
        p1: `tuple` line's first point [pix, pix]
        p2: `tuple` line's second point [pix, pix]
        color: `tuple` lines' color RGB [B, G, R] [int]
        thickness: `int` lines' thickness [pix]
        Dl: `int` distance in pixels between every point
    Returns:
        src: `cv2.mat` image with doted line drawn
    """

    # Get a number of intermediate points
    segments = discrete_contour((p1, p2), Dl)

    # Draw doted line 
    for segment in segments:
        cv2.circle(src, segment, thickness, color, -1) 

    # Return result
    return src

def discrete_contour(contour, Dl):

    """  Takes contour points to get a number of intermediate points
    Args:
        contour: `List` contour or list of points to get intermediate points
        Dl: `int` distance to get a point by segment
    Returns:
        new_contour: `List` new contour with intermediate points
    """

    # If contour has less of two points is not valid for operations
    if len(contour) < 2:
        print("Error: no valid segment")
        return contour

    # New contour variable
    new_contour = []

    # Iterate through all contour points
    for idx, cordinate in enumerate(contour):

        # Select next contour for operation
        if not idx == len(contour)-1:
            next_cordinate = contour[idx+1]
        else:
            next_cordinate = contour[0]

        # Calculate length of segment
        segment_lenth = math.sqrt((next_cordinate[0] - cordinate[0])**2 +\
                        (next_cordinate[1] - cordinate[1])**2)
        
        divitions = segment_lenth/Dl # Number of new point for current segment
        dy = next_cordinate[1] - cordinate[1]    # Segment's height
        dx = next_cordinate[0] - cordinate[0]    # Segment's width
        
        if not divitions:
            ddy = 0 # Dy value to sum in Y axis
            ddx = 0 # Dx value to sum in X axis
        else:
            ddy = dy/divitions  # Dy value to sum in Y axis
            ddx = dx/divitions  # Dx value to sum in X axis
        
        # get new intermediate points in segments
        for idx in range(0, int(divitions)):
            new_contour.append((int(cordinate[0] + (ddx*idx)), 
                                int(cordinate[1] + (ddy*idx))))    

    # Return new contour with intermediate points
    return new_contour

def get_projection_point_dst(coords_src, M):

    """  Gets the coordinate equivalent in surface projection space from original 
         view space 
    Args:
        coords_src: `numpy.darray`  coordinate in the original image space
        M: `numpy.darray` rotation matrix 
    Returns:
        coords_src: `numpy.darray`  projected coordinate in original view space
    """

    coords_dst = np.matmul(M, coords_src)
    coords_dst = coords_dst / coords_dst[2]
    coords_dst = [int(coords_dst[0]), int(coords_dst[1])]

    return coords_dst

def get_projection_point_src(coords_dst, INVM):

    """  Gets the coordinate equivalent in original view space from surface 
         projection space
    Args:
        coords_src: `numpy.darray`  coordinate in the original image space
        INVM: `numpy.darray` inverse rotation matrix 
    Returns:
        coords_src: `numpy.darray`  projected coordinate in original view space
    """

    # Calculate new coordinate
    coords_src = np.matmul(INVM, coords_dst)
    coords_src = coords_src / coords_src[2]
    coords_src = int(coords_src[0]), int(coords_src[1])

    return coords_src

def overlay_image(l_img, s_img, pos, transparency):

    """ Overlay 's_img on' top of 'l_img' at the position specified by
        pos and blend using 'alpha_mask' and 'transparency'.
    Args:
        l_img: `cv2.mat` inferior image to overlay superior image
        s_img: `cv2.mat` superior image to overlay
        pos: `tuple`  position to overlay superior image [pix, pix]
        transparency: `float` transparency in overlayed image
    Returns:
        l_img: `cv2.mat` original image with s_img overlayed
    """

    # Get superior image dimensions
    s_img_height, s_img_width, s_img_channels = s_img.shape

    if s_img_channels == 3 and transparency != 1:
        s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2BGRA)
        s_img_channels = 4

    # Take 3rd channel of 'img_overlay' image to get shapes
    img_overlay= s_img[:, :, 0:4]

    # cords assignation to overlay image 
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(l_img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(l_img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], l_img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], l_img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return l_img

    if s_img_channels == 4:
        # Get alphas channel
        alpha_mask = (s_img[:, :, 3] / 255.0) * transparency
        alpha_s = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_l = (1.0 - alpha_s)

        # Do the overlay with alpha channel
        for c in range(0, l_img.shape[2]):
            l_img[y1:y2, x1:x2, c] = (alpha_s * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_l * l_img[y1:y2, x1:x2, c])

    elif s_img_channels < 4:
        # Do the overlay with no alpha channel
        if l_img.shape[2] == s_img.shape[2]:
            l_img[y1:y2, x1:x2] = s_img[y1o:y2o, x1o:x2o]
        else:
            print("Error: to overlay images should have the same color channels")
            return l_img

    # Return results
    return l_img


# -----------------------------------------------------------------------------
# CAMERA CALIBRATION FUNCTIONS - CAMERA CALIBRATION FUNCTIONS - CAMERA CALIBRAT
def calibrate_camera(folder_path, calibration_file, n_x=6, n_y=9, 
    show_drawings=False):
    
    """ From chess board images find the coefficient and distortion of 
        the camera and save the matrix and distortion coefficients
    Args:
        folder_path: `string` Folder path with chessboard images
        calibration_file: `string` Name of the file (calibration matrix) to save or load parameters
        n_x: `int` Number of vertical divisions in chessboard
        n_y: `int` Number of horizontal divisions in chessboard
        show_drawings: `boolean` Enable/Disable show chessboard detections
    Returns:
        mtx: `numpy.narray` camera's distortion matrix
        dist: `numpy.narray` camera's distortion vector
    """

    mtx = dist = None

    # setup object points variables
    objp = np.zeros((n_y * n_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1, 2)
    image_points = []
    object_points = []

    # termination criteria for corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if not os.path.isdir(folder_path): 
        print(bcolors.FAIL + "[ERROR]: Folder doesn't exist" + bcolors.ENDC)
        return mtx, dist

    print("\nCalibration files \nFolder:{}\n".format(folder_path))
    img_list = os.listdir(folder_path)
    for idx, img_list_path in enumerate(img_list): # filelist[:] makes a copy of filelist.
        if not(img_list_path.endswith(".png")) and not(img_list_path.endswith(".jpg")):
            print(bcolors.FAIL + "\t{} - No image file \t- {}".format(idx, img_list_path) + bcolors.ENDC)
            continue

        img_src = cv2.imread(filename = os.path.join(folder_path, img_list_path))

        # Convert image to gray space
        img_src_gray = cv2.cvtColor(
            code = cv2.COLOR_RGB2GRAY,
            src = img_src) 

        # Find corners in chess board
        found, corners = cv2.findChessboardCorners(
            patternSize = (n_x, n_y),
            image = img_src_gray) 

        # If chess board was found in image
        if found: 
            
            if show_drawings:
                # Shows chess board corner detection Draw chess boards in images
                img_chess = cv2.drawChessboardCorners(
                    patternSize = (n_x, n_y), 
                    patternWasFound = found,
                    corners = corners,
                    image = img_src)
                cv2.imshow('Chess Board detected', img_chess); cv2.waitKey(0) 
                   
            # make fine adjustments to the corners so higher precision can
            # be obtained before appending them to the list
            corners2 = cv2.cornerSubPix(
                image = img_src_gray, 
                corners = corners, 
                winSize = (11, 11), 
                zeroZone = (-1, -1), 
                criteria = criteria) 

            # Include chess board corners to list
            image_points.append(corners2)

            # Include chess board objects to list
            object_points.append(objp)  

            print(bcolors.OKGREEN + "\t{} - Pattern found \t- {}".format(idx, img_list_path) + bcolors.ENDC)

        else:  
            print(bcolors.FAIL + "\t{} - No pattern found \t- {}".format(idx, img_list_path) + bcolors.ENDC)
    print()

    if len(image_points) and len(object_points):

        # perform camera calibration
        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(
                imageSize = img_src_gray.shape[::-1], 
                objectPoints = object_points,
                imagePoints = image_points, 
                cameraMatrix = None, 
                distCoeffs = None)
        print(bcolors.OKBLUE + "[INFO]: Camera calibration performed" + bcolors.ENDC)

        # Find optimum camera matrix to Undistorted images
        h,  w = img_src_gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            newImgSize = (w, h),
            imageSize = (w, h),
            cameraMatrix = mtx, 
            distCoeffs = dist, 
            alpha = 1)

        # Save camera parameters in specified path
        np.savez(os.path.join(folder_path, calibration_file), 
            newcameramtx = newcameramtx, 
            rvecs = rvecs, 
            tvecs = tvecs, 
            dist = dist, 
            ret = ret, 
            mtx = mtx, 
            roi = roi)

        print(bcolors.OKBLUE + "[INFO]: Camera calibration saved" + bcolors.ENDC)
        
    else:
        print(bcolors.WARNING + "[WARNING]: No camera calibration performed" + bcolors.ENDC)

    # Return results
    return mtx, dist

def load_camera_calibration(folder_path, calibration_file):

    """ Load matrix and distortion coefficients from file of camera calibration
    Args:
        folder_path: `string` Folder path with chessboard images
        calibration_file: `string` Name of the file (calibration matrix)
    Returns:
        mtx: `numpy.narray` camera's distortion matrix
        dist: `numpy.narray` camera's distortion vector
    """

    mtx = dist = None

    file_path = os.path.join(folder_path, calibration_file)
    if os.path.isfile(file_path):
        npzfile = np.load(file_path)
        dist = npzfile["dist"]
        mtx = npzfile["mtx"] 
        print(bcolors.OKBLUE + "[INFO]: Camera calibration loaded from {}".format(file_path) + bcolors.ENDC)
    
    else:
        print(bcolors.FAIL + "[ERROR]: No calibration {} file in folder {}".format(calibration_file, folder_path) + bcolors.ENDC)
    
    return mtx, dist


# -----------------------------------------------------------------------------
# COLOR THRESHOLDING FUNCTIONS - COLOR THRESHOLDING FUNCTIONS - COLOR THRESHOLD
def color_range_tunner(img_src, conf_file_path, tune=True,  
    space_model=cv2.COLOR_BGR2HSV):

    """ creates a window to tune with input image 'img_src' parameters of a
        color space model, then saves the configuration parameters in a npz 
        file to used later, function also returns parameters if tune mode is
        off
    Args:
        img_src: `cv2.math` input image to tune parameters of color space model 
        conf_file_path: `string` npz configuration file to save or load parameters
        tune: `boolean` Enable/Disable tuner mode
        space_model: `cv2.flag` Color space for OpenCV interface
    Returns:
        Arg1Min: `int` Minimum value of first argument in color model (HSV = H / HLS = H)
        Arg2Min: `int` Minimum value of second argument in color model (HSV = S / HLS = L)
        Arg3Min: `int` Minimum value of third argument in color model (HSV = V / HLS = S)
        Arg1Max: `int` Maximum value of first argument in color model (HSV = H / HLS = H)
        Arg2Max: `int` Maximum value of second argument in color model (HSV = S / HLS = L)
        Arg3Max: `int` Maximum value of third argument in color model (HSV = V / HLS = S)
    """

    # First values assignation
    Arg1Min = Arg2Min = Arg3Min = Arg1Max = Arg2Max = Arg3Max = 0

    # Read saved configuration
    if os.path.isfile(conf_file_path):
        npzfile = np.load(conf_file_path)
        Arg1Min = npzfile["Arg1Min"]; Arg2Min = npzfile["Arg2Min"]; Arg3Min = npzfile["Arg3Min"] 
        Arg1Max = npzfile["Arg1Max"]; Arg2Max = npzfile["Arg2Max"]; Arg3Max = npzfile["Arg3Max"]  
        print(bcolors.OKBLUE + "[INFO]: Color parameters loaded from {}".format(conf_file_path) + bcolors.ENDC)
    else: 
        print(bcolors.WARNING + "\t[WARNING]: No configuration file" + bcolors.ENDC)

    # Return parameters if not tune
    if not tune:
        return Arg1Min, Arg2Min, Arg3Min, Arg1Max, Arg2Max, Arg3Max

    if space_model == cv2.COLOR_BGR2HSV:
        win_name = "HSV_tunner_{}".format(conf_file_path)
        args = ["Hmin", "Hmax", "Smin", "Smax", "Vmin", "Vmax"]
    elif space_model == cv2.COLOR_BGR2HLS:
        win_name = "HLS_tunner_{}".format(conf_file_path)
        args = ["Hmin", "Hmax", "Lmin", "Lmax", "Smin", "Smax"]
    else:
        return Arg1Min, Arg2Min, Arg3Min, Arg1Max, Arg2Max, Arg3Max

    # Create tuner window
    cv2.namedWindow(win_name)

    # Set track bars to window
    cv2.createTrackbar(args[0], win_name ,Arg1Min, 255, nothing)
    cv2.createTrackbar(args[1], win_name ,Arg1Max, 255, nothing)
    cv2.createTrackbar(args[2], win_name ,Arg2Min, 255, nothing)
    cv2.createTrackbar(args[3], win_name ,Arg2Max, 255, nothing)
    cv2.createTrackbar(args[4], win_name ,Arg3Min, 255, nothing)
    cv2.createTrackbar(args[5], win_name ,Arg3Max, 255, nothing)

    # Create copy of input image
    img_aux = img_src.copy()

    uinput = '_'
    while not (uinput == ord('q') or uinput == ord('Q')):

        # Get trackbars position
        Arg1Min = cv2.getTrackbarPos(args[0], win_name)
        Arg1Max = cv2.getTrackbarPos(args[1], win_name)
        Arg2Min = cv2.getTrackbarPos(args[2], win_name)
        Arg2Max = cv2.getTrackbarPos(args[3], win_name)
        Arg3Min = cv2.getTrackbarPos(args[4], win_name)
        Arg3Max = cv2.getTrackbarPos(args[5], win_name)

        # Set thresholds
        COLOR_LANE_LINE_MIN = np.array([Arg1Min, Arg2Min, Arg3Min],np.uint8)     
        COLOR_LANE_LINE_MAX = np.array([Arg1Max, Arg2Max, Arg3Max],np.uint8)

        # Convert and thresh 
        img_color = cv2.cvtColor(img_aux, space_model)
        mask = cv2.inRange(img_color, COLOR_LANE_LINE_MIN, COLOR_LANE_LINE_MAX)

        # Show result
        result_img = np.concatenate((img_aux, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), axis = 1)
        result_img = cv2.resize(result_img, (int(result_img.shape[1]*0.5), int(result_img.shape[0]*0.5)))
        cv2.putText(img = result_img, text = conf_file_path, org = (10, 20), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 3, color = (0, 0, 0))
        cv2.putText(img = result_img, text = conf_file_path, org = (10, 20), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 1, color = (0, 255, 255))
        cv2.imshow(win_name, result_img)
        
        # Read user input
        uinput = cv2.waitKey(30) & 0xFF

    # Destroy created windows
    cv2.destroyWindow(win_name)

    # Save configuration in file
    np.savez_compressed(
        conf_file_path, 
        Arg1Min  = Arg1Min, Arg2Min = Arg2Min, Arg3Min = Arg3Min, 
        Arg1Max  = Arg1Max, Arg2Max = Arg2Max, Arg3Max = Arg3Max)
    print("\t[INFO]: New {} configuration saved".format(conf_file_path))
    
    # Return tunned parameters
    return Arg1Min, Arg2Min, Arg3Min, Arg1Max, Arg2Max, Arg3Max

def find_lanelines(img_src, COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL, 
                   VERT_TRESH=0.6, FILT_KERN=5):

    """ Finds a simple linear regression canny edge detection for each lane line (Right and Left)
    Args:
        img_src: `cv2.math` input image to find and approximate lane lines
        COLOR_TRESH_MIN: `list` Minimum parameters to color thresholding
        COLOR_TRESH_MAX: `list` Maximum parameters to color thresholding
        COLOR_MODEL: `list` Color space in cv2 interface to color thresholding
        VERT_TRESH: `float` Normalized value to ignore vertical image values
        FILT_KERN: `int` (odd) size/kernel of filter (Bilateral)
    Returns:
        Lm: `float`  linear regression slope of left lane line
        Lb: `float`  linear regression y-intercept of left lane line
        Rm: `float`  linear regression slope of right lane line
        Rb: `float`  linear regression y-intercept of right lane line
        Left_Lines: `list` list of left lines with which left lane line was calculated
        Right_Lines: `list` list of left lines with which right lane line was calculated
    """

    # -------------------------------------------------------------------------
    # GET BINARY MASK - GET BINARY MASK - GET BINARY MASK - GET BINARY MASK - G

    mask = get_binary_mask(
        COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
        COLOR_TRESH_MAX = COLOR_TRESH_MAX, 
        COLOR_MODEL = COLOR_MODEL,
        VERT_TRESH = VERT_TRESH, 
        FILT_KERN = FILT_KERN,
        img_src = img_src)

    # Get canny image
    mask_canny = cv2.Canny(image = mask, threshold1 = 0, threshold2 = 255)
    # cv2.imshow("mask_canny", mask_canny); cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # IDENTIFY RIGHT AND LEFT LANE LINES - IDENTIFY RIGHT AND LEFT LANE LINES -

    # Get lane lines
    lane_lines = cv2.HoughLinesP(
            image = mask_canny, 
            rho = 0.2, 
            theta = np.pi/180.,
            lines = 100, 
            threshold = 10,     # (10) The minimum number of intersections to detect a line 
            minLineLength = 10, # (10) The minimum number of points that can form a line. 
                                # Lines with less than this number of points are disregarded.
            maxLineGap = 5)     # (5) The maximum gap between two points to be considered in the same line.

    # to draw and show results
    # for line in lane_lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_src, (x1, y1), (x2, y2),(0, 0, 255), 2)
    # cv2.imshow("HoughLinesP", img_src)

    # Declaration of useful variables
    Left_Lines = []; Right_Lines = []
    bottom_line = ((0, img_src.shape[0]), (img_src.shape[1], img_src.shape[0]))
    Lllx = []; Llly = []; Rllx = []; Rlly = []
    
    # Apply some heuristics to remove some detected lane lines
    for line in lane_lines:
        for x1, y1, x2, y2 in line:
            angle = math.atan2(y2 - y1, x2 - x1)*(180.0 / math.pi)
            interc = line_intersection(bottom_line, ((x1, y1), (x2, y2)))

            # Conditions for left lines
            COND1 = angle < -10 and angle > -80
            COND2 = x1 < img_src.shape[1]*0.5 or x2 < img_src.shape[1]*0.5
            COND3 = interc[0] > 0

            # Conditions for right lines
            COND4 = int(angle) > 10
            COND5 = x1 > img_src.shape[1]*0.5 or x2 > img_src.shape[1]*0.5
            COND6 = interc[0] > 0 and interc[0] < img_src.shape[1]

            # For left lines
            if COND1 and COND2 and COND3:
                Left_Lines.append(line)
                cv2.line(mask, (x1, y1), (x2, y2),(255, 200, 0), 2)
                Lllx.append(x1); Lllx.append(x2); 
                Llly.append(y1); Llly.append(y2); 

            # For right lines
            elif COND4 and COND5 and COND6:
                Right_Lines.append(line)
                cv2.line(mask, (x1, y1), (x2, y2),(77, 195, 255), 2)
                Rllx.append(x1); Rllx.append(x2); 
                Rlly.append(y1); Rlly.append(y2); 

    # Find linear regression to approximate each lane line    
    Lm = None; Lb = None
    Rm = None; Rb = None

    # Calculate simple linear regression for left lane line
    if len(Llly) and len(Lllx):
        Lm, Lb = np.polyfit(x = Lllx, y = Llly, deg = 1, rcond=None, full=False, w=None, cov=False)

    # Calculate simple linear regression for right lane line
    if len(Rllx) and len(Rlly):
        Rm, Rb = np.polyfit(x = Rllx, y = Rlly, deg = 1, rcond=None, full=False, w=None, cov=False)

    # # to draw and show results
    # for line in Right_Lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_src, (x1, y1), (x2, y2),(220, 280, 0), 2)
    # for line in Left_Lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_src, (x1, y1), (x2, y2),(255, 0, 255), 2)
    # cv2.imshow("HoughLinesP_Heuricstic", img_src)

    # Return results
    return Lm, Lb, Rm, Rb, Left_Lines, Right_Lines

def draw_lanelines(img_src, Right_Lane_Line, Left_Lane_Line, VERT_TRESH=0.6, 
                   draw_lines=True, draw_regression=True, draw_info=True):

    """  Draws lane lines in image
    Args:
        img_src: `cv2.math` input image to draw lane lines
        Right_Lane_Line: `LaneLine` Right lane line
        Left_Lane_Line: `LaneLine` Right lane line
        VERT_TRESH: `float` normalized vertical value to start printing lines
        draw_lines: `boolean` input Enable/Disable line printings
        draw_regression: `boolean` Enable/Disable linear regression printings
        draw_info: `boolean` Enable/Disable information printings
    Returns:
        img_src: `cv2.math` input image with lane lines drawn
    """

    # Create a copy of input image
    img_foreground = img_src.copy()

    # If draw line regressions 
    if draw_regression:

        Ly1 = Ry1 = int(img_src.shape[0]*VERT_TRESH)
        Ly2 = Ry2 = img_src.shape[0]

        Rm = Right_Lane_Line.regression["m"]
        Rb = Right_Lane_Line.regression["b"]
        Lm = Left_Lane_Line.regression["m"]
        Lb = Left_Lane_Line.regression["b"]

        line_thickness = 3

        if Lm is not None and Lb is not None:
            Lx1 = int((Ly1 - Lb)/Lm); Lx2 = int((Ly2 - Lb)/Lm)
            line_color = (0, 0, 255) if Left_Lane_Line.lost else (0, 255, 0)
            cv2.line(img_foreground, (Lx1, Ly1), (Lx2, Ly2), line_color, line_thickness)

        if Rm is not None and Rb is not None:
            Rx1 = int((Ry1 - Rb)/Rm); Rx2 = int((Ry2 - Rb)/Rm)
            line_color = (0, 0, 255) if Right_Lane_Line.lost else (0, 255, 0)
            cv2.line(img_foreground, (Rx1, Ry1), (Rx2, Ry2), line_color, line_thickness)

    # If draw individual lines
    if draw_lines:
        for line in Right_Lane_Line.lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_foreground, (x1, y1), (x2, y2),(255, 0, 0), 2)
        for line in Left_Lane_Line.lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_foreground, (x1, y1), (x2, y2),(255, 0, 0), 2)

    # Overlay image with lines drawn over original image with transparency
    img_src = cv2.addWeighted(
        src2 = img_foreground, 
        src1 = img_src, 
        alpha = 0.7, 
        beta = 0.3, 
        gamma = 0)

     # If print lane lines information in image
    if draw_info:
        str_list = ["Left Line:", 
                    "   Line: {}".format(len(Left_Lane_Line.lines)),
                    "   lost: {} ({})".format(Left_Lane_Line.lost, Left_Lane_Line.lost_frames),
                    "   m: {}".format(round(Left_Lane_Line.regression["m"], 2)),
                    "   b: {}".format(round(Left_Lane_Line.regression["b"], 2)),
                    " ",
                    "Right Line:", 
                    "   Line: {}".format(len(Right_Lane_Line.lines)),
                    "   lost: {} ({})".format(Right_Lane_Line.lost, Right_Lane_Line.lost_frames),
                    "   m: {}".format(round(Right_Lane_Line.regression["m"], 2)),
                    "   b: {}".format(round(Right_Lane_Line.regression["b"], 2))]
          
        print_list_text(img_src = img_src, str_list = str_list, thickness = 1, origin = (10, 20))

    # Return result
    return img_src

def get_binary_mask(img_src, COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL, 
    VERT_TRESH, FILT_KERN):

    """ Get a binary mask from color space models thresholds
    Args:
        img_src: `cv2.math` input image to get binary mask from color thresholding
        COLOR_TRESH_MIN: `list` Minimum parameters to color thresholding
        COLOR_TRESH_MAX: `list` Maximum parameters to color thresholding
        COLOR_MODEL: `list` Color space in cv2 interface to color thresholding
        VERT_TRESH: `float` Normalized value to ignore vertical image values
        FILT_KERN: `int` (odd) size/kernel of filter (Bilateral)
    Returns:
        mask: `cv2.math` binary mask with color thresholding process
    """

    # Apply convolution filter to smooth the image
    img_filt = cv2.bilateralFilter(
        src = img_src, d = FILT_KERN, sigmaColor = FILT_KERN, sigmaSpace = FILT_KERN)
    # cv2.imshow("image_filtered", img_filt); cv2.waitKey(0)

    # Crete a mask/ Binary image to find the lane lines
    mask = np.zeros((img_filt.shape[0], img_filt.shape[1], 1), dtype=np.uint8)
    for idx in range(0, len(COLOR_MODEL)):
        
        # Convert images to hsv (hue, saturation, value) color space
        img_tresh = cv2.cvtColor(src = img_filt.copy(), code = COLOR_MODEL[idx])
        img_tresh[0:int(img_tresh.shape[0]*VERT_TRESH),:] = [0, 0, 0]

        # Thresh by color for yellow lane lines
        img_tresh = cv2.inRange(
            src = img_tresh, 
            lowerb = COLOR_TRESH_MIN[idx], 
            upperb = COLOR_TRESH_MAX[idx])

        # Combine masks with OR operation
        mask = cv2.bitwise_or(mask, img_tresh)
    # cv2.imshow("binary_mask", mask); cv2.waitKey(0)

    return mask

def load_color_spaces_ranges(img, files_list, tune=False):

    """ Load color spaces parameters
    Args:
        img: `cv2.math` image to calibrate color space parameters
        files_list: `list` [string] list with color space models configuration (.npz)
        tune: `boolean` enable/disable parameters tunning
    Returns:
        COLOR_TRESH_MIN: `list` Minimum parameters to color thresholding
        COLOR_TRESH_MAX: `list` Maximum parameters to color thresholding
        COLOR_MODEL: `list` Color space in cv2 interface to color thresholding
    """

    COLOR_TRESH_MIN = []
    COLOR_TRESH_MAX = []
    COLOR_MODEL = []

    for file_name in files_list:
        
        # Extract color space model from file name
        color_extension = file_name.split(".")[-2].split("_")[-1]
        color_space = cv2.COLOR_BGR2HSV if color_extension == "hsv" else cv2.COLOR_BGR2HLS
        
        # Get color parameters from file
        Arg1min, Arg2min, Arg3min, Arg1max, Arg2max, Arg3max = color_range_tunner(
        img_src = img, tune = tune, conf_file_path = file_name, space_model = color_space)

        # Prepare maximum and minimum parameters
        COLOR_MIN = np.array([Arg1min, Arg2min, Arg3min], np.uint8)
        COLOR_MAX = np.array([Arg1max, Arg2max, Arg3max], np.uint8)

        # Add features to list
        COLOR_TRESH_MIN.append(COLOR_MIN)
        COLOR_TRESH_MAX.append(COLOR_MAX)
        COLOR_MODEL.append(color_space)

    return COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL


# -----------------------------------------------------------------------------
# COLOR THRESHOLDING FUNCITONS - COLOR THRESHOLDING FUNCITONS - COLOR THRESHOLD
def find_projection(Lm, Lb, Rm, Rb, ORIGINAL_SIZE, UNWARPED_SIZE, 
    HozTop = 50, HozBottom = 0, porc = 0.3):

    """ Find projection surface parameters from lane lines
    Args:
        Lm: `float`  linear regression slope of left lane line
        Lb: `float`  linear regression y-intercept of left lane line
        Rm: `float`  linear regression slope of right lane line
        Rb: `float`  linear regression y-intercept of right lane line
        ORIGINAL_SIZE: `tuple` original size (width, height)
        UNWARPED_SIZE: `tuple` Unwarped size (width, height)
        HozTop: `int` Superior threshold value
        HozBottom: `int` Superior threshold value
        porc: `float` percentage to adjust surface's geometry in UNWARPED_SIZE
    Returns:
        M: `numpy.darray` transformation matrix 
        INVM: `numpy.darray` inverse of transformation matrix 
        src_points: `np.array` original size (p1, p2, p3, p4) [pix]
        dst_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        size_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        vp: `tuple` vanishing point (x, y) [pix]
    """ 

    # Get coordinates X and Y from linear regressions
    Ly1 = Ry1 = 0
    Ly2 = Ry2 = ORIGINAL_SIZE[1]
    Lx1 = int((Ly1 - Lb)/Lm); Lx2 = int((Ly2 - Lb)/Lm)
    Rx1 = int((Ry1 - Rb)/Rm); Rx2 = int((Ry2 - Rb)/Rm)

    # Declaration of lines
    LL = ((Lx1, Ly1), (Lx2, Ly2)) # Left line
    RL = ((Rx1, Ry1), (Rx2, Ry2)) # Right line

    # Calculate intersection (vanishing point) between left and right lane lines
    vp = line_intersection(line1 = LL, line2 = RL)

    # Top and bottom line declaration
    TL = ((0, vp[1] + HozTop), (ORIGINAL_SIZE[0], vp[1] + HozTop)) # Top Line
    BL = ((0, ORIGINAL_SIZE[1] - HozBottom), # Bottom Line
          (ORIGINAL_SIZE[0], ORIGINAL_SIZE[1] - HozBottom))

    # Points of geometry projection
    p1 = line_intersection(line1 = LL, line2 = TL) # Top Left intersection
    p2 = line_intersection(line1 = RL, line2 = TL) # Top Right intersection
    p3 = line_intersection(line1 = RL, line2 = BL) # Bottom Right intersection
    p4 = line_intersection(line1 = LL, line2 = BL) # Bottom Left intersection

    # Center pattern with camera center
    hoz_disp = int(vp[0] - ORIGINAL_SIZE[0]*0.5)
    p1[0] += hoz_disp; p2[0] += hoz_disp; p3[0] += hoz_disp; p4[0] += hoz_disp

    # Source points
    src_points = np.array([p1, p2, p3, p4], dtype=np.float32) 

    # Destination points
    dst_points = np.array([
        [int(UNWARPED_SIZE[0]*porc), 0], 
        [UNWARPED_SIZE[0] - int(UNWARPED_SIZE[0]*porc), 0], 
        [UNWARPED_SIZE[0] - int(UNWARPED_SIZE[0]*porc), UNWARPED_SIZE[1]],
        [int(UNWARPED_SIZE[0]*porc), UNWARPED_SIZE[1]]
        ], dtype=np.float32)

    # Calculate projection matrix 
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Print information
    spacer = 33
    print(bcolors.OKBLUE + "[INFO]: Perspective transform performed\n" + bcolors.ENDC)
    print("\t"+"-"*spacer); print("\t| Source \t| Destination\t|"); print("\t"+"-"*spacer)
    for pt_src, pt_dst in zip(src_points.astype(int), dst_points.astype(int)):
        print("\t| {},\t {} \t| {},\t {}\t|".format(pt_src[0], pt_src[1], pt_dst[0], pt_dst[1]))
    print("\t" + "-"*spacer + "\n")

    # Calculate inverse of projection matrix 
    INVM = np.linalg.inv(M)

    # Points of surface projection corners in original source 
    pp1 = get_projection_point_dst((0, 0, 1), INVM)
    pp2 = get_projection_point_dst((UNWARPED_SIZE[0], 0, 1), INVM)
    pp3 = get_projection_point_dst((UNWARPED_SIZE[0], UNWARPED_SIZE[1], 1), INVM)
    pp4 = get_projection_point_dst((0, UNWARPED_SIZE[1], 1), INVM)
    size_points = np.array([pp1, pp2, pp3, pp4], dtype=np.float32) 

    # Return results
    return M, INVM, src_points, dst_points, size_points, vp

def load_camera_projection(folder_path, projection_file):

    """ Load camera projection parameters from file
    Args:
        folder_path: `string` Folder path where to save projection parameters
        projection_file: `string` projection parameters file name
    Returns:
        M: `numpy.darray` transformation matrix 
        INVM: `numpy.darray` inverse of transformation matrix 
        src_points: `np.array` original size (p1, p2, p3, p4) [pix]
        dst_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        size_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        vp: `tuple` vanishing point (x, y) [pix]
        xm_per_pix: `float` [m/pix] horizontal pixel to meters relation
        ym_per_pix: `float` [m/pix] vertical pixel to meters relation
    """

    # First assignation 
    M = INVM = src_points = dst_points = size_points = vp = None

    # Load projection parameters
    file_path = os.path.join(folder_path, projection_file)
    if os.path.isfile(file_path):
        npzfile = np.load(file_path)
        size_points = npzfile["size_points"] 
        xm_per_pix = npzfile["xm_per_pix"]
        ym_per_pix = npzfile["ym_per_pix"]
        src_points = npzfile["src_points"] 
        dst_points = npzfile["dst_points"] 
        INVM = npzfile["INVM"]
        vp = npzfile["vp"]
        M = npzfile["M"]
        print(bcolors.OKBLUE + "[INFO]: Projection parameters loaded from {}".format(file_path) + bcolors.ENDC)
    
    else:
        print(bcolors.FAIL + "[ERROR]: No projection {} file in folder {}".format(projection_file, folder_path) + bcolors.ENDC)
    
    # Return loaded parameters
    return M, INVM, src_points, dst_points, size_points, vp, xm_per_pix, ym_per_pix

def save_camera_projection(folder_path, projection_file, size_points, src_points,
    dst_points, INVM, M, vp, xm_per_pix, ym_per_pix):

    """ save camera projection parameters from file
    Args:
        folder_path: `string` Folder path to save projection parameters
        projection_file: `string` projection parameters file name
        M: `numpy.darray` transformation matrix 
        INVM: `numpy.darray` inverse of transformation matrix 
        src_points: `np.array` original size (p1, p2, p3, p4) [pix]
        dst_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        size_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        vp: `tuple` vanishing point (x, y) [pix]
        xm_per_pix: `float` [m/pix] horizontal pixel to meters relation
        ym_per_pix: `float` [m/pix] vertical pixel to meters relation
    Returns:

    """

    # Save camera parameters in specified path
    np.savez(os.path.join(folder_path, projection_file), 
        xm_per_pix = xm_per_pix,
        ym_per_pix = ym_per_pix,
        size_points = size_points,
        src_points = src_points,
        dst_points = dst_points,
        INVM = INVM,
        vp = vp,
        M = M)

def draw_projection_parameters(img_src, img_proj, UNWARPED_SIZE, M, src_points, 
    dst_points, size_points, vp, img_src_text=(), img_pro_text=(), 
    draw_inner_projection=True, draw_inner_geoemtry=True, draw_outer_geoemtry=True):
    
    """ Draw projection geometries in original and projection image
    Args:
        img_src: `cv2.math` input image to draw projection parameters
        UNWARPED_SIZE: `tuple` Unwarped size (width, height) [pix]
        M: `numpy.darray` transformation matrix 
        src_points: `np.array` source points (p1, p2, p3, p4) [pix]
        dst_points: `np.array` src_points in projection space (p1, p2, p3, p4) [pix]
        size_points: `np.array` surface projection size points (p1, p2, p3, p4) [pix]
        vp: `tuple` vanishing point (x, y) [pix]
        img_src_text: `list` list with text to print in original image source
        img_pro_text: `list` list with text to print in projection image
        draw_inner_projection: `boolean` Enable/Disable inner projection drawing
        draw_inner_geoemtry: `boolean` Enable/Disable inner geometry drawing
        draw_outer_geoemtry: `boolean` Enable/Disable outer geometry drawing
    Returns:
        img_src: `cv2.math` input image with projection parameters drawn
    """

    # Create copy of input image
    img_src_cop = img_src.copy()

    if draw_inner_projection:
        # Draw road surface projection over original image
        cv2.drawContours(
            image = img_src, contours = [src_points.astype(int)], 
                contourIdx = 0, color = (0, 255, 0), thickness = -1)

        # Overlay road surface projection over original image
        img_src = cv2.addWeighted(
            src1 = img_src_cop, src2 = img_src, 
            alpha = 0.7, beta = 0.3, gamma = 0)

    thickness = 2 # Lines thickness
    if draw_outer_geoemtry:
        dotline(src = img_src, p1 = tuple(vp), p2 = tuple(src_points[0]), 
                color = (0, 255, 255), thickness = thickness, Dl = 10)
        dotline(src = img_src, p1 = tuple(vp), p2 = tuple(src_points[1]), 
                color = (0, 255, 255), thickness = thickness, Dl = 10)
        dotline(src = img_src, p1 = (0, vp[1]), p2 = (img_src.shape[1], vp[1]), 
                color = (0, 255, 255), thickness = thickness, Dl = 10)

    if draw_outer_geoemtry:
        cv2.line(img_src, tuple(size_points[0]), tuple(size_points[1]), (0, 255, 0), thickness)
        cv2.line(img_src, tuple(size_points[1]), tuple(size_points[2]), (0, 255, 0), thickness)
        cv2.line(img_src, tuple(size_points[2]), tuple(size_points[3]), (0, 255, 0), thickness)
        cv2.line(img_src, tuple(size_points[3]), tuple(size_points[0]), (0, 255, 0), thickness)

    if draw_inner_geoemtry:
        cv2.line(img_src, tuple(src_points[0]), tuple(src_points[3]), (0, 0, 255), thickness)
        cv2.line(img_src, tuple(src_points[1]), tuple(src_points[2]), (0, 0, 255), thickness)

    # Draw vanishing point
    if draw_outer_geoemtry:
        cv2.circle(img_src, tuple(vp), 8, (0, 0, 0), -1) 
        cv2.circle(img_src, tuple(vp), 5, (0, 255, 255), -1) 
        print_list_text(
            img_src = img_src, str_list = ("vp", ""), origin = (int(vp[0]+10), int(vp[1] -10)), 
            color = (0, 255, 255), thickness = 1, fontScale = 0.8)

    # Draw points and lines of surface projection
    if draw_inner_geoemtry:
        for idx, pt in enumerate(src_points):
            cv2.circle(img_src, tuple(pt), 8, (0, 0, 0), -1) 
            cv2.circle(img_src, tuple(pt), 5, (0, 0, 255), -1) 
            print_list_text(
                img_src = img_src, str_list = ("p{}".format(idx), ""), origin = (int(pt[0]+10), int(pt[1] -10)), 
                color = (0, 0, 255), thickness = 1, fontScale = 0.8)
    
    # Draw geometry of surface projection in original image
    if draw_outer_geoemtry:
        for idx, pt in enumerate(size_points):
            cv2.circle(img_src, tuple(pt), 8, (0, 0, 0), -1) 
            cv2.circle(img_src, tuple(pt), 5, (0, 255, 0), -1) 
            print_list_text(
            img_src = img_src, str_list = ("psz{}".format(idx), ""), origin = (int(pt[0]+10), int(pt[1])), 
            color = (0, 255, 0), thickness = 1, fontScale = 0.8)

    # Resize images
    img_src = cv2.resize(
        dsize = (int(img_src.shape[1]*0.75), int(img_src.shape[0]*0.75)),
        src = img_src)
    img_proj = cv2.resize(src = img_proj, dsize = (300, img_src.shape[0]))

    print_list_text(
        img_src = img_src, str_list = img_src_text, y_space = 30,
        origin = (10, 20), color = (255, 255, 255), thickness = 1, fontScale = 0.7)
    print_list_text(
        img_src = img_proj, str_list = img_pro_text, y_space = 25,
        origin = (10, 20), color = (0, 255, 0), thickness = 1, fontScale = 0.7)

    # Concatenate results
    img_src = np.concatenate((img_src, img_proj), axis = 1)

    return img_src


# -----------------------------------------------------------------------------
# PROJECTION AND POLY FIT FUNCTIONS - PROJECTION AND POLY FIT FUNCTIONS - PROJE
def fit_polynomial(src_warped, binary_warped, nwindows=9, margin=100, minpix=50,
    left_fit=None, right_fit=None, pp3=None, pp4=None):

    """ Get polynomial regression for left and right lane line
    Args:
        src_warped: `cv2.math` input original projection image to find lane lines regression
        binary_warped: `cv2.math` input binary projection image to find lane lines regression
        nwindows: `int` Number of sliding windows
        margin: `int` width of the windows +/- margin
        minpix: `int` minimum number of pixels found to recenter window
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        pp3: `tuple` right inferior surface projection corner
        pp4: `tuple` left inferior surface projection corner
    Returns:
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        leftx: `numpy.ndarray` left lane line x coordinates 
        lefty: `numpy.ndarray` left lane line y coordinates 
        rightx: `numpy.ndarray` right lane line x coordinates 
        righty: `numpy.ndarray` right lane line y coordinates 
        out_img: `cv2.math` binary mask image with linear regression and windows drawings
    """

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(
        nwindows = nwindows, margin = margin, minpix = minpix,
        binary_warped = binary_warped, draw_windows = True,
        src_warped = src_warped, left_fit=left_fit, right_fit=right_fit,
        pp3 = pp3, pp4 = pp4)

    # Fit a second order polynomial to each using `np.polyfit`
    if len(lefty) and len(leftx):
        if len(lefty) == len(leftx):
            if len(lefty) > 10:
                left_fit = np.polyfit(lefty, leftx, 2)
    if len(righty) and len(rightx):
        if len(righty) == len(rightx):
            if len(righty) > 10:
                right_fit = np.polyfit(righty, rightx, 2)

    # Return regressions of left and right lane lines
    return left_fit, right_fit, leftx, lefty, rightx, righty, out_img

def draw_lane_projections(src_img, left_fit, right_fit, leftx, lefty, 
    rightx, righty, print_warning = False, color=(0, 255, 255)):

    """ description
    Args:
        src_img: `cv2.math` image to draw lane lines projections and other results
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        leftx: `numpy.ndarray` left lane line x coordinates 
        lefty: `numpy.ndarray` left lane line y coordinates 
        rightx: `numpy.ndarray` right lane line x coordinates 
        righty: `numpy.ndarray` right lane line y coordinates 
        print_warning: `boolean` enable/disable warning printings
    Returns:
        surface_geometry: `list` list of coordinates with road surface projection
        src_img: `cv2.math` input image with lane lines projections and other results drawn
    """

    # Generate x and y values for plotting
    ploty = np.linspace(0, src_img.shape[0]-1, src_img.shape[0] )
    right_fitx = left_fitx = []
    try:
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        else:
            if print_warning:
                print(bcolors.WARNING + '[WARNING]: The function failed to fit a left line!' + bcolors.ENDC)
            overlay_image(src_img, cv2.resize(warning_icon, (160, 50)), (40, 20), 1)
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        else:
            if print_warning:
                print(bcolors.WARNING + '[WARNING]: The function failed to fit a right line!' + bcolors.ENDC)
            overlay_image(src_img, cv2.resize(warning_icon, (160, 50)), (src_img.shape[1]-200, 20), 1)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print(bcolors.FAIL + '[ERROR]: The function failed to fit a line!' + bcolors.ENDC)
        overlay_image(src_img, cv2.resize(critica_icon, (160, 50)), (src_img.shape[1]-200, src_img.shape[0]-100), 1)
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    surface_geometry = []

    for y, x in enumerate(right_fitx):
        cv2.circle(img = src_img, center = (int(x), int(y)), radius = 5, 
            color = color, thickness = -1) 
        surface_geometry.append((int(x), int(y)))
    surface_geometry.reverse()

    # Plots the left and right polynomials on the lane lines
    for y, x in enumerate(left_fitx):
        cv2.circle(img = src_img, center = (int(x), int(y)), radius = 5, 
        color = color, thickness = -1) 
        surface_geometry.append((int(x), int(y)))

    # Colors in the left and right lane regions
    src_img[lefty, leftx] = [255, 0, 0]     # Print left lane line in blue
    src_img[righty, rightx] = [0, 0, 255]   # Print right lane line in red

    return surface_geometry, src_img

def find_lane_pixels(src_warped, binary_warped, nwindows=9, margin=100, minpix=50, 
    draw_windows=True, vert_tresh_zero=0.8, left_fit=None, right_fit=None, pp3=None,
    pp4=None, draw_histograms=False):

    """ Get a binary mask from color space models thresholds
    Args:
        src_warped: `cv2.math` input original projection image to find lane lines regression
        binary_warped: `cv2.math` input projection image to find lane lines
        nwindows: `int` Number of sliding windows
        margin: `int` width of the windows +/- margin
        minpix: `int` minimum number of pixels found to recenter window
        draw_windows: `boolean` Enable/Disable windows drawings
        vert_tresh_zero: `float` Normalized value to ignore vertical image values
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        pp3: `tuple` right inferior surface projection corner
        pp4: `tuple` left inferior surface projection corner
        draw_histograms: `boolean` enable/disable process and histograms printings
    Returns:
        leftx: `numpy.ndarray` x coordinates for linear regression of left lane line 
        lefty: `numpy.ndarray` y coordinates for linear regression of left lane line 
        rightx: `numpy.ndarray` x coordinates for linear regression of right lane line 
        righty: `numpy.ndarray` y coordinates for linear regression of right lane line
        out_img: `cv2.math` binary mask with windows for linear regression drawn
    """

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Concert original source image in HSV color space model
    src_warped_hsv = cv2.cvtColor(src_warped, cv2.COLOR_BGR2HSV)

    if pp3 is None and pp4 is None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2) # Middle point in input image

        # To search close to polynomial regression
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    else:
        leftx_base = int(pp4[0])
        rightx_base = int(pp3[0])

    # Set height of windows - based on n windows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Current positions to be updated later for each window in n windows
    leftx_current = leftx_base
    rightx_current = rightx_base

    leftx  = []; lefty  = []
    rightx = []; righty = []

    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = src_warped.shape[0] - (window+1) * window_height
        win_y_high = src_warped.shape[0] - window*window_height
        
        # Recenter window with polinomial fit
        y = int(win_y_low + (win_y_high-win_y_low)*0.5)
        if left_fit is not None: 
            A, B, C = left_fit
            leftx_current = int(A*(y**2) + B*y + C)
        if right_fit is not None: 
            A, B, C = right_fit
            rightx_current = int(A*(y**2) + B*y + C)

        # Find the four below boundaries of the window ###
        win_xleft_low   = leftx_current  - margin
        win_xleft_high  = leftx_current  + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Calculate window width
        window_widht = win_xright_high - win_xright_low

        if win_xleft_low<0: win_xleft_low = 0
        if win_xleft_high<0: win_xleft_high = 0
        if win_xright_low>src_warped.shape[1]: win_xright_low = src_warped.shape[1]
        if win_xright_high>src_warped.shape[1]: win_xright_high = src_warped.shape[1]

        # Draw the windows on the visualization image
        if draw_windows:
            cv2.rectangle(out_img,(win_xleft_low, win_y_low),
            (win_xleft_high,win_y_high),(255,0, 255), 2) 
            cv2.rectangle(out_img,(win_xright_low, win_y_low),
            (win_xright_high, win_y_high), (255, 0, 255), 2) 

        # ---------------------------------------------------------------------
        # Get the histograms and their maximum arguments
        roi_img_r = src_warped_hsv[win_y_low:win_y_high, win_xright_low:win_xright_high]
        histr = cv2.calcHist([roi_img_r],[2],None,[256],[0,256])
        roi_img_l = src_warped_hsv[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
        histl = cv2.calcHist([roi_img_l],[2],None,[256],[0,256])
        histr[0]=0;histl[0]=0 # To ignore black pixels
        idx_max_r = np.argmax(histr)
        idx_max_l = np.argmax(histl)

        # Calculate first binary mask from histograms thresholds
        thresh_offset_low_r = 0 
        thresh_offset_high_r = 50
        roi_img_binary_r = cv2.inRange(
            src = roi_img_r, 
            lowerb = np.array([0, thresh_offset_low_r, idx_max_r+thresh_offset_high_r]), 
            upperb = np.array([255, 255, 255]))
        thresh_offset_low_l = 0 
        thresh_offset_high_l = 50
        roi_img_binary_l = cv2.inRange(
            src = roi_img_l, 
            lowerb = np.array([0, thresh_offset_low_l, idx_max_l+thresh_offset_high_l]), 
            upperb = np.array([255, 255, 255]))

        # If first window ignore part of window where vehicles appears
        if not window:
            if roi_img_binary_l is not None:
                roi_img_binary_l[int(roi_img_binary_l.shape[0]*vert_tresh_zero):, :] = 0
            if roi_img_binary_r is not None:
                roi_img_binary_r[int(roi_img_binary_r.shape[0]*vert_tresh_zero):, :] = 0

        # Identify the x and y positions of all nonzero pixels in the image
        source_l = source_r = "H"
        
        nonzeroxl = nonzeroyl = nonzeroxl = []
        if roi_img_binary_l is not None:
            mask_l = roi_img_binary_l.copy()
            nonzerol  = mask_l.nonzero()
            mask_l_src = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
            nonzero_porc_l = len(nonzerol[0])/(window_widht*window_height)
            if not len(nonzerol[0]): 
                    mask_l = binary_warped[
                        win_y_low:win_y_high, win_xleft_low:win_xleft_high]
                    nonzerol  = mask_l.nonzero()
                    source_l = "B"
            elif nonzero_porc_l < 0.05:
                mask_l = cv2.bitwise_or(
                    mask_l_src,
                    mask_l)
                nonzerol  = mask_l.nonzero()
                source_l = "H+B(or)"
            elif nonzero_porc_l > 0.20:
                mask_l = cv2.bitwise_and(
                    mask_l_src,
                    mask_l)
                nonzerol  = mask_l.nonzero()
                source_l = "H+B(and)"
            nonzeroyl = np.array(nonzerol[0]) + win_y_low 
            nonzeroxl = np.array(nonzerol[1]) + win_xleft_low 

        nonzeror = nonzeroyr = nonzeroxr = []
        if roi_img_binary_r is not None:
            mask_r = roi_img_binary_r.copy()
            nonzeror  = mask_r.nonzero()
            mask_r_src = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high]
            nonzero_porc_r = len(nonzeror[0])/(window_widht*window_height)
            if not len(nonzeror[0]): 
                mask_r = binary_warped[
                    win_y_low:win_y_high, win_xright_low:win_xright_high]
                nonzeror  = mask_r.nonzero()
                source_r = "B"
            elif nonzero_porc_r < 0.05:
                mask_r = cv2.bitwise_or(mask_r_src, mask_r)
                nonzeror  = mask_r.nonzero()
                source_r = "H+B(or)"
            elif nonzero_porc_r > 0.20:
                mask_r = cv2.bitwise_and(mask_r_src, mask_r)
                nonzeror  = mask_r.nonzero()
                source_r = "H+B(and)"
            nonzeroyr = np.array(nonzeror[0]) + win_y_low 
            nonzeroxr = np.array(nonzeror[1]) +  win_xright_low

        # Draw process setp by step
        if draw_histograms:
            
            for idx in range(len(nonzeroyr)):
                out_img[int(nonzeroyr[idx]), int(nonzeroxr[idx])] = [0, 0, 255]
            for idx in range(len(nonzeroyl)):
                out_img[int(nonzeroyl[idx]), int(nonzeroxl[idx])] = [0, 0, 255]

            print_list_text(out_img, 
                ("his:{}".format(idx_max_l), 
                 "por:{}%".format(round(nonzero_porc_l, 2)),
                 "src:{}".format(source_l),
                 "len:{}".format(len(nonzeroyl))), 
                origin = (int(win_xleft_low + (win_xleft_high - win_xleft_low)*0.5)+110, 
                          int(win_y_low + (win_y_high - win_y_low)*0.5)-25), 
                color = (0, 255, 0), thickness = 1, fontScale = 0.35, y_space=15)
            print_list_text(out_img, 
                ("his:{}".format(idx_max_r), 
                 "por:{}%".format(round(nonzero_porc_r, 2)),
                 "src:{}".format(source_r),
                 "len:{}".format(len(nonzeroyr))), 
                origin = (int(win_xright_low + (win_xright_high - win_xright_low)*0.5)+110, 
                          int(win_y_low + (win_y_high - win_y_low)*0.5)-25), 
                color = (0, 255, 0), thickness = 1, fontScale = 0.35, y_space=15)

            if roi_img_binary_l is not None:
                roi_img_binary_l = cv2.cvtColor(roi_img_binary_l, cv2.COLOR_GRAY2BGR)
                roi_img_binary_l[:,:,0] = 0; roi_img_binary_l[:,:,1] = 0
                show_img_1 = np.concatenate(
                    (src_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high], 
                    cv2.cvtColor(mask_l_src, cv2.COLOR_GRAY2BGR),
                    roi_img_binary_l), 
                    axis = 1)
                cv2.imshow("thresh_space_1", show_img_1)

            if roi_img_binary_r is not None:
                roi_img_binary_r = cv2.cvtColor(roi_img_binary_r, cv2.COLOR_GRAY2BGR)
                roi_img_binary_r[:,:,0] = 0; roi_img_binary_r[:,:,2] = 0
                show_img_2 = np.concatenate(
                    (src_warped[win_y_low:win_y_high, win_xright_low:win_xright_high], 
                    cv2.cvtColor(mask_r_src, cv2.COLOR_GRAY2BGR),
                    roi_img_binary_r), 
                    axis = 1)
                cv2.imshow("thresh_space_2", show_img_2)
        
            plt.plot(histr,color = 'r')
            plt.plot([idx_max_r, idx_max_r], [0, histr[idx_max_r]], marker = 'o', color = 'r')
            plt.plot([idx_max_r+thresh_offset_high_r, idx_max_r+thresh_offset_high_r], 
                    [0, histr[idx_max_r]], marker = 'o', color = 'r')
            plt.plot(histl,color = 'g')
            plt.plot([idx_max_l, idx_max_l], [0, histl[idx_max_l]], marker = 'o', color = 'g')
            plt.plot([idx_max_l+thresh_offset_high_l, idx_max_l+thresh_offset_high_l], 
                    [0, histl[idx_max_l]], marker = 'o', color = 'g')
            plt.xlim([0,256])
            plt.draw()
            plt.pause(0.01)
            plt.clf()

            cv2.imshow("out_img_proc", out_img)
            user_in = cv2.waitKey(0)

            # if user_in & 0xFF == ord('t') or user_in == ord('T'):
            #     load_color_spaces_ranges(
            #                 files_list = color_files_list, 
            #                 img = img_pro,
            #                 tune = True)

        # ---------------------------------------------------------------------
        # If you found > minpix pixels, recenter next window on their mean position
        if len(nonzeroyl) >= minpix and left_fit is None:
            if len(nonzeroxl) > 0:
                leftx_current = np.int(np.mean(nonzeroxl))

        if len(nonzeroyr) >= minpix and right_fit is None:     
            if len(nonzeroxr) > 0:
                rightx_current = np.int(np.mean(nonzeroxr))

        # Extract left and right line pixel positions
        if len(leftx) == len(lefty):
            leftx  = np.concatenate((leftx, nonzeroxl), axis = None)
            lefty  = np.concatenate((lefty, nonzeroyl), axis = None)

        if len(rightx) == len(righty):
            rightx = np.concatenate((rightx, nonzeroxr), axis = None)
            righty = np.concatenate((righty, nonzeroyr), axis = None)

    return leftx.astype(int), lefty.astype(int), rightx.astype(int), righty.astype(int), out_img

def find_pix_meter_relations(UNWARPED_SIZE, left_fit, right_fit, 
    x_distance_m=3.7, y_distance_m=3., pix_dashed_line=76.):

    """ Calculate vertical and horizontal pixel to meters relation
    Args:
        UNWARPED_SIZE: `tuple` Unwarped size (width, height)
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        x_distance_m: `float` [m] road width or distance between lane lines
        y_distance_m: `float` [m] length of lane line
        pix_dashed_line: `int` [pix] length of lane line
    Returns:
        xm_per_pix: `float` [m/pix] horizontal pixel to meters relation
        ym_per_pix: `float` [m/pix] vertical pixel to meters relation
    """

    # Calculate the width of road in pixels
    y = UNWARPED_SIZE[1] # Evaluate the bottom of image

    # Evaluate polynomial fit in y
    left_fitx = left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2]
    right_fitx = right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2]

    # Get distance bewteen lines in pixels
    x_distance_pix = abs(right_fitx - left_fitx) 

    # calcualte [m/pix] meters per pixel relation in x dimension
    xm_per_pix = x_distance_m/x_distance_pix 

    # calcualte [m/pix] meters per pixel relation in y dimension
    ym_per_pix = y_distance_m/pix_dashed_line 

    # Print information
    spacer = 33
    print(bcolors.OKBLUE + "[INFO]: Pixels relations calculated\n" + bcolors.ENDC)
    print("\t"+"-"*spacer); print("\t| xm_per_pix \t| ym_per_pix\t|"); print("\t"+"-"*spacer)
    print("\t| {}\t| {},\t\t|".format(round(xm_per_pix, 4), round(ym_per_pix, 4)))
    print("\t" + "-"*spacer + "\n")

    return xm_per_pix, ym_per_pix

def measure_curvatures(left_fit, right_fit, ym_per_pix=1., xm_per_pix=1.,  
    y_eval=0):

    """ Calculate left and right lane line curvature
    Args:
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        xm_per_pix: `float` [m/pix] horizontal pixel to meters relation
        ym_per_pix: `float` [m/pix] vertical pixel to meters relation
        y_eval: `int` value to evaluate curvature
    Returns:
        right_curvature: `float` [m] curvature of left lane line
        left_curvature: `float` [m] curvature of right lane line
    """

    # Varibles assignation
    left_curvature = right_curvature = 0

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image

    # Measure curvature for left lane line
    if left_fit is not None:

        Al = left_fit[0]*(xm_per_pix/(ym_per_pix**2))
        Bl = left_fit[1]*(xm_per_pix/ym_per_pix)

        # Calculation of R_curve (radius of curvature)
        left_curvature = ((1 + (2*Al*y_eval + Bl)**2)**1.5) / np.absolute(2*Al)
    
    # Measure curvature for right lane line
    if right_fit is not None:
        
        Ar = right_fit[0]*(xm_per_pix/(ym_per_pix**2))
        Br = right_fit[1]*(xm_per_pix/ym_per_pix)

        # Calculation of R_curve (radius of curvature)
        right_curvature = ((1 + (2*Ar*y_eval + Br)**2)**1.5) / np.absolute(2*Ar)

    return right_curvature, left_curvature

def get_car_road_position(left_fit, right_fit, xm_per_pix, UNWARPED_SIZE):

    """ Calculate position of the vehicle with respect to center of road
    Args:
        left_fit: `numpy.ndarray` second order linear regression of left lane line
        right_fit: `numpy.ndarray` second order linear regression of right lane line
        xm_per_pix: `float` [m/pix] vertical pixel to meters relation
        UNWARPED_SIZE: `tuple` Unwarped size (width, height)
    Returns:
        car_lane_pos: `float` [m] position of the vehicle with respect to center of road
    """

    if left_fit is None or right_fit is None: return None

    left_fitx  = left_fit[0]*UNWARPED_SIZE[1]**2 + left_fit[1]*UNWARPED_SIZE[1] + left_fit[2]
    right_fitx = right_fit[0]*UNWARPED_SIZE[1]**2 + right_fit[1]*UNWARPED_SIZE[1] + right_fit[2]
    x_distance_pix = int(left_fitx + abs(left_fitx - right_fitx)*0.5)

    car_lane_pos = x_distance_pix - UNWARPED_SIZE[0]*0.5
    car_lane_pos = car_lane_pos*xm_per_pix

    return car_lane_pos

def draw_results(img_src, img_pro, img_pro_mask, UNWARPED_SIZE, size_points, 
    src_points, dst_points, vp, M, INVM, surface_geometry, src_name ="", 
    right_curvature = 1, left_curvature = 1, car_lane_pos = 1):

    """ Draw results of surface projection, curvature and others
    Args:
        img_src: `cv2.math` DESCRIPTION
        img_pro: `cv2.math` DESCRIPTION
        img_pro_mask: `cv2.math` DESCRIPTION
        UNWARPED_SIZE: `tuple` Unwarped size (width, height)
        src_points: `np.array` original size (p1, p2, p3, p4) [pix]
        dst_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        size_points: `np.array` Unwarped size (p1, p2, p3, p4) [pix]
        vp: `tuple` vanishing point (x, y) [pix]
        M: `numpy.darray` transformation matrix 
        INVM: `numpy.darray` inverse of transformation matrix 
        surface_geometry: `list` list of coordinates with road surface projection
        src_name: `string` name of image source file
        right_curvature: `float` [m] curvature of left lane line
        left_curvature: `float` [m] curvature of right lane line
        car_lane_pos: `float` [m] position of the vehicle with respect to center 
    Returns:
        img_res: `cv2.math` image with drawings
    """

    img_res = img_src.copy()

    # Get perspective transformation
    img_proj = cv2.warpPerspective(
        dsize = UNWARPED_SIZE, 
        src = img_src, 
        M = M)

    # Draw central line in projection
    p_top = get_projection_point_dst((int(src_points[0][0]+(src_points[1][0] - src_points[0][0])*0.5), int(src_points[0][1]), 1), M)
    p_bottom = get_projection_point_dst((int(src_points[3][0]+(src_points[2][0] - src_points[3][0])*0.5), int(src_points[2][1]), 1), M)
    cv2.line(img_proj, tuple(p_bottom), tuple(p_top), (0, 0, 255), 3)

    # Select road side
    side = "left" if car_lane_pos > 0 else "right"

    # Prepare list of text to print
    img_src_text = (
        src_name, 
        "curvature (ave)= {} [m]".format(round(abs(left_curvature+right_curvature)*0.5, 2)),
        "Left curvature = {} [m]".format(round(left_curvature, 2)), 
        "Right curvature = {} [m]".format(round(right_curvature, 2)),  
        "vehicle is {} [m] {} of center".format(round(abs(car_lane_pos), 2), side) if car_lane_pos is not None else "Unknown")
    img_pro_text = ("sky_view", "")

    for idx, pt in enumerate(surface_geometry):
        surface_geometry[idx] = get_projection_point_src((pt[0], pt[1], 1), INVM)

    if len(surface_geometry):
        cv2.drawContours(
            contours = [np.array(surface_geometry)], 
            color = (0, 255, 0),
            image = img_res, 
            thickness = -1,
            contourIdx = 0)

    # Draw left line from linear regression
    for pt in surface_geometry[0: int(len(surface_geometry)*0.5)]:
        cv2.circle(img_res, tuple(pt), 6, (0, 0, 255), -1) 
        pt_proj = get_projection_point_dst((pt[0], pt[1], 1), M)
        cv2.circle(img_proj, tuple(pt_proj), 6, (0, 0, 255), -1) 
    # Draw left line from linear regression
    for pt in surface_geometry[int(len(surface_geometry)*0.5):]:
        cv2.circle(img_res, tuple(pt), 6, (255, 0, 0), -1) 
        pt_proj = get_projection_point_dst((pt[0], pt[1], 1), M)
        cv2.circle(img_proj, tuple(pt_proj), 6, (255, 0, 0), -1) 

    # Overlay road surface projection over original image
    img_src = cv2.addWeighted(
        src1 = img_res, 
        src2 = img_src, 
        alpha = 0.3, 
        gamma = 0.0,
        beta = 1.0)

    # Draw surface projection parameters
    img_res = draw_projection_parameters(
        UNWARPED_SIZE = UNWARPED_SIZE, 
        draw_inner_projection = False, 
        draw_inner_geoemtry = False, 
        draw_outer_geoemtry = True,
        img_src_text = img_src_text, 
        img_pro_text = img_pro_text,
        size_points = size_points,
        img_proj = img_proj,
        img_src = img_src.copy(),
        src_points = src_points, 
        dst_points = dst_points, 
        vp = vp, 
        M = M)

    img_proj_mask = cv2.resize(
        dsize = (300, img_res.shape[0]),
        interpolation = cv2.INTER_AREA,
        src = img_pro_mask)
    img_res = np.concatenate((img_res, img_proj_mask), axis = 1)

    return img_res


# =============================================================================
# MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # HYPER PARAMETERS - HYPER PARAMETERS - HYPER PARAMETERS - HYPER PARAMETERS 
    
    # process parameters    
    folder_dir_image = "./test_images"  # folder with images 
    folder_dir_video = "./test_videos"  # folder with videos 
    out_folder_image = "./output_images" # Output folder for images
    out_folder_video = "./output_videos" # Output folder for videos
    img_list = os.listdir(folder_dir_image)     # Images list
    video_list = os.listdir(folder_dir_video)   # Videos list
    
    results_window_name = "surface_projection_result"
    show_process_calibration = True # Show process for camera calibration
    show_process_SurfaceProj = True # Sow process for surface projection
    show_process_images = True  # Show process for images
    show_process_videos = True # Show process for videos
    Save_results = True # Enable/Disable results saving

    # Variables for camera calibration
    cam_calibration_folder = "./camera_cal" # Folder path with chessboard images
    cam_calibration_file = "cam_calibration.npz" # File name to load/save calibration
    chessboard_vert_div = 6 # Number of vertical divisions in chessboard
    chessboard_hori_div = 9 # Number of horizontal divisions in chessboard

    # Color Thresholding Parameters
    Tune_ranges = False # Enable/Disable parameters tuning
    color_files_list = [
            './lane_lines_conf_hls.npz',
            './white_conf_hsv.npz',
            './yellow_conf_hsv.npz']

    # Projection Parameters
    projection_file = "projection_params.npz"
    UNWARPED_SIZE = (1280, 720) # Unwarped size (width, height)
    VERT_TRESH = 0.6 # Normalized value to ignore vertical image values
    HozBottom = 0 # Superior threshold value
    HozTop = 30 # Superior threshold value
    porc = 0.3 # percentage to adjust surface's geometry in UNWARPED_SIZE

    # Parameters for pixel relation
    pix_dashed_line = 50. # [pix] length of lane line
    x_distance_m = 3.7 # [m] road width or distance between lane lines
    y_distance_m = 3. # [m] length of discontinuous lane line
    
    # Poly Fit parameters
    nwindows = 9 # Number of sliding windows
    margin = 100 # width of the windows +/- margin
    minpix = 10  # minimum number of pixels found to recenter window

    # -------------------------------------------------------------------------
    # CAMERA CALIBRATION - CAMERA CALIBRATION - CAMERA CALIBRATION - CAMERA CAL
    """
    1.  Briefly state how you computed the camera matrix and distortion coefficients. 
        Provide an example of a distortion corrected calibration image.
    """
    if show_process_calibration:

        # Perform camera calibration
        mtx, dist = calibrate_camera(
            calibration_file = cam_calibration_file,
            folder_path = cam_calibration_folder, 
            n_x = chessboard_hori_div, 
            n_y = chessboard_vert_div,
            show_drawings = False)

        # Load camera calibration from file
        mtx, dist = load_camera_calibration(
            calibration_file = cam_calibration_file,
            folder_path = cam_calibration_folder)

        # Confirm and see results of camera calibration
        img_scr = cv2.imread(os.path.join(cam_calibration_folder, "calibration1.jpg"))
        img_scr_undistort = cv2.undistort(img_scr, mtx, dist)
        img_scr = print_list_text( img_src = img_scr, str_list = ("ORIGINAL IMAGE", ""), 
                origin = (10, 50), color = (0, 0, 255), thickness = 5, fontScale = 1.5)
        img_scr_undistort = print_list_text( 
                img_src = img_scr_undistort, str_list = ("UNDISTORTED IMAGE", ""), 
                origin = (10, 50), color = (0, 255, 0), thickness = 5, fontScale = 1.5)
        img_scr = np.concatenate((img_scr, img_scr_undistort), axis = 1)
        img_scr = cv2.resize(
            dsize = (int(img_scr.shape[1]*0.5), int(img_scr.shape[0]*0.5)),
            src = img_scr)
        cv2.imwrite(
            filename = os.path.join(out_folder_image, "cam_calibration.jpg"), 
            img = img_scr)
        win_name = "Camera_calibration_result"
        cv2.imshow(win_name, img_scr)
        print("[INFO]: Press any key to continue"); cv2.waitKey(0)
        cv2.destroyWindow(win_name)
        print()

    # -------------------------------------------------------------------------
    # SURFACE PROJECTION - SURFACE PROJECTION - SURFACE PROJECTION - SURFACE PR
    if show_process_SurfaceProj:

        """
        2.  Describe how (and identify where in your code) you used color transforms, 
            gradients or other methods to create a thresholded binary image. Provide 
            an example of a binary image result.
        """

        # Tuning/Reading color space model parameters
        img_proj_ref = "straight_lines1.jpg"
        img_src = cv2.imread(os.path.join(folder_dir_image, img_proj_ref))

        COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL = load_color_spaces_ranges(
            files_list = color_files_list, 
            tune = Tune_ranges,
            img = img_src)

        # ---------------------------------------------------------------------
        """
        3.  Describe how (and identify where in your code) you performed a perspective 
            transform and provide an example of a transformed image.
        """

        # Find lane lines in image
        Lm, Lb, Rm, Rb, Left_Lines, Right_Lines = find_lanelines(
                img_src = img_src.copy(),
                COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
                COLOR_TRESH_MAX = COLOR_TRESH_MAX, 
                COLOR_MODEL = COLOR_MODEL,  
                VERT_TRESH = VERT_TRESH, 
                FILT_KERN = 20)

        # Find projection from lane lines detection 
        M, INVM, src_points, dst_points, size_points, vp = find_projection(
            ORIGINAL_SIZE = (img_src.shape[1], img_src.shape[0]),
            UNWARPED_SIZE = UNWARPED_SIZE, 
            Lm = Lm, Lb = Lb, Rm = Rm, Rb = Rb, 
            HozBottom = HozBottom,
            HozTop = HozTop, 
            porc = porc)

        # re-calculation of Normalized value to ignore vertical image values
        VERT_TRESH = float(vp[1])/float(img_src.shape[0])

        # ---------------------------------------------------------------------
        """
        4.  Describe how (and identify where in your code) you identified lane-line
            pixels and fit their positions with a polynomial?
        """

        # Get perspective transformation
        img_proj = cv2.warpPerspective(
            dsize = UNWARPED_SIZE, 
            src = img_src, 
            M = M)

        # Get a binary mask from color space models thresholds
        img_proj_mask = get_binary_mask(
            COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
            COLOR_TRESH_MAX = COLOR_TRESH_MAX, 
            COLOR_MODEL = COLOR_MODEL, 
            img_src = img_proj, 
            VERT_TRESH = 0, 
            FILT_KERN= 20)

        # Get polynomial regression for left and right lane line
        left_fit, right_fit, leftx, lefty, rightx, righty, img_proj_mask = fit_polynomial(
            binary_warped = img_proj_mask,
            src_warped = img_proj,
            nwindows = nwindows, 
            margin = margin, 
            minpix = minpix)

        # ---------------------------------------------------------------------
        """
        5.  Describe how (and identify where in your code) you calculated the radius
            of curvature of the lane and the position of the vehicle with respect to 
            center.
        """

        # Calculate vertical and horizontal pixel to meters relation
        xm_per_pix, ym_per_pix = find_pix_meter_relations(
            UNWARPED_SIZE = UNWARPED_SIZE,
            right_fit = right_fit,
            left_fit = left_fit,
            pix_dashed_line = pix_dashed_line,
            x_distance_m = x_distance_m,
            y_distance_m = y_distance_m)

        # Calculate left and right lane line curvature
        right_curvature, left_curvature = measure_curvatures(
            y_eval = UNWARPED_SIZE[1]*ym_per_pix,
            ym_per_pix = ym_per_pix, 
            xm_per_pix = xm_per_pix, 
            right_fit = right_fit, 
            left_fit = left_fit)

        # Calculate position of the vehicle with respect to center of road
        car_lane_pos = get_car_road_position(
            UNWARPED_SIZE = UNWARPED_SIZE,
            xm_per_pix = xm_per_pix, 
            right_fit = right_fit,
            left_fit = left_fit)

        # ---------------------------------------------------------------------
        """
        6.  Provide an example image of your result plotted back down onto the road 
            such that the lane area is identified clearly.
        """

        # Draw lane lines projections
        surface_geometry, img_proj_mask = draw_lane_projections(
            src_img = img_proj_mask, 
            left_fit=left_fit, 
            right_fit=right_fit,
            rightx=rightx, 
            righty=righty,
            leftx=leftx, 
            lefty=lefty)

        # Draw results of surface projection, curvature and others
        img_res = draw_results(
                surface_geometry = surface_geometry,
                right_curvature = right_curvature,
                left_curvature = left_curvature,
                car_lane_pos = car_lane_pos,
                UNWARPED_SIZE = UNWARPED_SIZE, 
                img_pro_mask = img_proj_mask,
                src_name =  img_proj_ref,
                size_points = size_points, 
                src_points = src_points, 
                dst_points = dst_points,
                img_src = img_src, 
                img_pro = img_src, 
                INVM = INVM,
                vp = vp, 
                M = M)
        cv2.imshow(results_window_name, img_res)
        print("[INFO]: Press any key to continue"); cv2.waitKey(0)

        # Save surface projection parameters
        save_camera_projection(
            folder_path = cam_calibration_folder, 
            projection_file = projection_file, 
            size_points = size_points, 
            src_points = src_points,
            dst_points = dst_points, 
            INVM = INVM, 
            M = M, 
            vp = vp, 
            xm_per_pix = xm_per_pix, 
            ym_per_pix = ym_per_pix)
    else:

        COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL = load_color_spaces_ranges(
            files_list = color_files_list, 
            img = None)

        # Load projection parameters
        M, INVM, src_points, dst_points, size_points, vp , xm_per_pix, ym_per_pix = \
            load_camera_projection(
                folder_path = cam_calibration_folder, 
                projection_file =projection_file)

        print()

    # -------------------------------------------------------------------------
    # IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - I
    if show_process_images:

        options = "\nImage Options:" \
            + "\n\tA: Continue with next image" \
            + "\n\tT: Tune Color ranges"
        print(options)

        # Read every images in folder 
        for idx in range(0, len(img_list)):

            # Read image
            img_src = cv2.imread(os.path.join(folder_dir_image, img_list[idx]))

            # Get perspective transformation
            img_pro = cv2.warpPerspective(
                dsize = UNWARPED_SIZE,
                src = img_src,  
                M = M)

            # Get a binary mask from color space models thresholds
            img_pro_mask = get_binary_mask(
                COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
                COLOR_TRESH_MAX = COLOR_TRESH_MAX, 
                COLOR_MODEL = COLOR_MODEL,
                img_src = img_pro,  
                VERT_TRESH = 0, 
                FILT_KERN= 5)

            # Get polynomial regression for left and right lane line
            left_fit, right_fit, leftx, lefty, rightx, righty, img_proj_mask = fit_polynomial(
                binary_warped = img_pro_mask,
                src_warped = img_pro,
                nwindows = nwindows, 
                margin = margin, 
                minpix = minpix,
                pp3 = dst_points[2],
                pp4 = dst_points[3])

            # Draw lane lines projections
            surface_geometry, img_proj_mask = draw_lane_projections(
                src_img = img_proj_mask, 
                left_fit=left_fit, 
                right_fit=right_fit,
                rightx=rightx, 
                righty=righty,
                leftx=leftx, 
                lefty=lefty)

            # # Calculate left and right lane line curvature
            right_curvature, left_curvature = measure_curvatures(
                y_eval = UNWARPED_SIZE[1]*ym_per_pix,
                ym_per_pix = ym_per_pix, 
                xm_per_pix = xm_per_pix, 
                right_fit = right_fit, 
                left_fit = left_fit)

            # # Calculate position of the vehicle with respect to center of road
            car_lane_pos = get_car_road_position(
                UNWARPED_SIZE = UNWARPED_SIZE,
                xm_per_pix = xm_per_pix, 
                right_fit = right_fit, 
                left_fit = left_fit)

            # Draw results of surface projection, curvature and others
            img_res = draw_results(
                surface_geometry = surface_geometry,
                right_curvature = right_curvature,
                left_curvature = left_curvature,
                car_lane_pos = car_lane_pos,
                UNWARPED_SIZE = UNWARPED_SIZE, 
                img_pro_mask = img_proj_mask,
                src_name =  img_list[idx],
                size_points = size_points, 
                src_points = src_points, 
                dst_points = dst_points,
                img_src = img_src, 
                img_pro = img_src, 
                INVM = INVM,
                vp = vp, 
                M = M)

            # Show results
            cv2.imshow(results_window_name, img_res)
            while True:
                user_in = cv2.waitKey(10) 
                if user_in & 0xFF == ord('a') or user_in == ord('A'): break
                if user_in & 0xFF == ord('t') or user_in == ord('t'): 
                    COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL = load_color_spaces_ranges(
                        files_list = color_files_list, 
                        img = img_pro,
                        tune = True)

            # Write result image
            if Save_results:
                cv2.imwrite(
                    filename = os.path.join(out_folder_image, img_list[idx]), 
                    img = img_res)

        print()

    # -------------------------------------------------------------------------
    # VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - V
    if show_process_videos:

        options = "\nVideo Options:" \
            + "\n\tQ: Exit" \
            + "\n\tN: Continue with next video" \
            + "\n\tA: Stop/Reproduce Video" \
            + "\n\tT: Tune Color ranges" \
            + "\n" 
        print(options)

        # Process in video list
        for idx in range(0, len(video_list)): 

            # Create and initialize Lane lines variables
            Right_Lane_Line = Line("Right_Lane_Line")
            Left_Lane_Line = Line("Left_Lane_Line")

            # Variables for video recording
            if Save_results:
                fps = 30. # Frames per second

                # For mp4 video - Codec H264, format MP4
                # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
                # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                fourcc = cv2.VideoWriter_fourcc(*'X264')
                # fourcc = cv2.cv.CV_FOURCC(*'H264')
                # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')

                video_format = ".mp4"
                # video_format = ".avi" 

                name_1 = os.path.join(out_folder_video, video_list[idx].split(".")[-2]+video_format) # File name and format
                if 'video_out' in locals(): del video_out
                        
            # Start video capture
            cap = cv2.VideoCapture(os.path.join(folder_dir_video, video_list[idx]))
            reproduce = True

            # While video capture
            while(cap.isOpened()):

                # If reproduction on or pause off
                if reproduce:
                
                    # Read frame in video capture
                    ret, img_src = cap.read()
                    if not ret: break

                    # Get perspective transformation
                    img_pro = cv2.warpPerspective(
                        dsize = UNWARPED_SIZE,
                        src = img_src, 
                        M = M)
                    
                    # Get a binary mask from color space models thresholds
                    img_pro_mask = get_binary_mask(
                        COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
                        COLOR_TRESH_MAX = COLOR_TRESH_MAX, 
                        COLOR_MODEL = COLOR_MODEL, 
                        img_src = img_pro, 
                        VERT_TRESH = 0, 
                        FILT_KERN= 0)

                    # Get polynomial regression for left and right lane line
                    left_fit, right_fit, leftx, lefty, rightx, righty, img_proj_mask = \
                        fit_polynomial(
                            binary_warped = img_pro_mask,
                            src_warped = img_pro,
                            nwindows = nwindows, 
                            margin = margin, 
                            minpix = minpix,
                            left_fit = Left_Lane_Line.current_fit, 
                            right_fit = Right_Lane_Line.current_fit,
                            pp3 = dst_points[2],
                            pp4 = dst_points[3])

                    # Assign new polynomial fits to lane lines
                    Left_Lane_Line.assing_fit(
                        y_eval = UNWARPED_SIZE[1],
                        poly_fit = np.asarray(left_fit), 
                        x_coords = leftx,
                        y_coords = lefty)
                    Right_Lane_Line.assing_fit(
                        y_eval = UNWARPED_SIZE[1],
                        poly_fit = np.asarray(right_fit),
                        x_coords = leftx,
                        y_coords = lefty)

                    # Draw lane lines projections
                    surface_geometry, img_proj_mask = draw_lane_projections(
                        src_img = img_proj_mask, 
                        left_fit=Left_Lane_Line.current_fit, 
                        right_fit=Right_Lane_Line.current_fit,
                        rightx=rightx, 
                        righty=righty,
                        leftx=leftx, 
                        lefty=lefty)
                    _, _ = draw_lane_projections(
                        src_img = img_proj_mask, 
                        color = (0, 255, 0),
                        right_fit=right_fit,
                        left_fit=left_fit, 
                        rightx=[], 
                        righty=[],
                        leftx=[], 
                        lefty=[])

                    # Calculate left and right lane line curvature
                    right_curvature, left_curvature = measure_curvatures(
                        y_eval = UNWARPED_SIZE[1]*ym_per_pix,
                        ym_per_pix = ym_per_pix, 
                        xm_per_pix = xm_per_pix, 
                        right_fit = Right_Lane_Line.current_fit, 
                        left_fit = Left_Lane_Line.current_fit)
                    Left_Lane_Line.radius_of_curvature = left_curvature
                    Right_Lane_Line.radius_of_curvature = right_curvature

                    # Calculate position of the vehicle with respect to center of road
                    car_lane_pos = get_car_road_position(
                        UNWARPED_SIZE = UNWARPED_SIZE,
                        xm_per_pix = xm_per_pix, 
                        right_fit = Right_Lane_Line.current_fit, 
                        left_fit = Left_Lane_Line.current_fit)
            
                    # Draw results of surface projection, curvature and others
                    img_res = draw_results(
                        surface_geometry = surface_geometry,
                        right_curvature = right_curvature,
                        left_curvature = left_curvature,
                        car_lane_pos = car_lane_pos,
                        UNWARPED_SIZE = UNWARPED_SIZE, 
                        img_pro_mask = img_proj_mask,
                        src_name =  video_list[idx],
                        size_points = size_points, 
                        src_points = src_points, 
                        dst_points = dst_points,
                        img_src = img_src, 
                        img_pro = img_pro, 
                        INVM = INVM,
                        vp = vp, 
                        M = M)
                    cv2.imshow(results_window_name, img_res)

                    # Write to video file
                    if Save_results: 
                        if not 'video_out' in locals():
                            video_out = cv2.VideoWriter(
                                frameSize = (img_res.shape[1], img_res.shape[0]),
                                filename = name_1, 
                                fourcc = fourcc, 
                                fps = fps)
                        video_out.write(img_res) 

                # Wait user input
                user_in = cv2.waitKey(10) 
                if user_in & 0xFF == ord('q') or user_in == ord('Q'): exit()
                if user_in & 0xFF == ord('n') or user_in == ord('N'): break
                if user_in & 0xFF == ord('a') or user_in == ord('A'): 
                    reproduce = not reproduce
                if user_in & 0xFF == ord('t') or user_in == ord('T'): 
                    COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL = load_color_spaces_ranges(
                        files_list = color_files_list, 
                        img = img_pro,
                        tune = True)

        # Destroy video variables
        cap.release(); cv2.destroyAllWindows(); print()


# =============================================================================
#                     (: (: SORRY FOR MY ENGLISH! :) :)