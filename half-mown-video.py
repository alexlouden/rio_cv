#-------------------------------------------------------------------------
# Name:		Mown grass edge detector
# Purpose:	   Detect and display the edge between cut and uncut regions of grass
#
# Author:	  Alex Louden
#
# Created:	 15/11/2012
# Copyright:   (c) Alex 2012
#
# Requirements:	matplotlib, numpy, OpenCV
#-------------------------------------------------------------------------

import os
import cv2
import math
import cPickle
import matplotlib.pyplot as plt
import numpy as np

# OpenCV uses 0-180 for hue values - green is ~50
min_green = np.array([25, 15, 50])
max_green = np.array([70, 256, 256])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

frame_size = (640, 360)

# Line detection tolerance
tolerance = 35

big_contour_size = 500

# Maximum change in position and angle (pix/frame and deg/frame)
max_delta_theta = math.pi / 15
max_delta_rho = 30

# Number of frames to use for determing most likely line
num_last_frames_for_frequency = 5


def main(filename):
    # Load video
    video = cv2.VideoCapture(filename)

    codecid = "XVID"
    codec = cv2.cv.FOURCC(*codecid)
    video_out = cv2.VideoWriter(
        filename.split('.')[0] + '_out.avi', codec, 10, frame_size)

    os.chdir('.\\Out')

    # Check OpenCV was able to open the video
    if not video.isOpened():
        raise RuntimeError('Input file not open')

    if not video_out.isOpened():
        raise RuntimeError('Output file not open')

##	width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
##	height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
##	print height, width

##	frame_lines = []

    line_rho = 308
    line_theta = 0

    # First 500 frames
    for framecount in range(300):
        f, img = video.read()

        # Reached end of video
        if not f:
            return

##		if framecount < 78:
##			continue
##		elif framecount > 78:
##			return

        print "=" * 50
        print "Processing frame {}".format(framecount)

        img = cv2.resize(img, frame_size)
        vis, lines = do_green_line_detection(img, framecount)

        # Make angles between -theta and theta
        lines = [(-rho, theta - math.pi) if theta > math.pi / 2 else (rho, theta)
                 for rho, theta in lines]

        print "lines:", lines

        poss_lines = filter(lambda x:
                            abs(x[1] - line_theta) <= max_delta_theta
                            and
                            abs(x[0] - line_rho) <= max_delta_rho,
                            lines)

        print "possible lines:", poss_lines

        line_thickness = 1

        if poss_lines:
            rhos, thetas = zip(*poss_lines)
            num = len(poss_lines)
            line_theta = sum(thetas) / num
            line_rho = sum(rhos) / num
            line_thickness = 5

        print "current line:", (line_rho, line_theta)

        pt1, pt2 = polar_to_rect(line_rho, line_theta)
        cv2.line(vis, pt1, pt2, (255, 0, 0), line_thickness)

        # cv2.imwrite('out_frame_{}.png'.format(framecount), vis)
        video_out.write(vis)

##		# Store lines in list
##		frame_lines.append(lines)


##		freq_lines = get_frequent_lines(frame_lines, framecount)
##		print freq_lines

##	print frame_lines

##def get_frequent_lines(frame_lines, framenum):
##
##	# Get most frequent line in last several frames
##	last_lines = frame_lines[
##		max(framenum-num_last_frames_for_frequency,0) : framenum]
##
##	# Flatten list
##	last_lines = [i for l in last_lines for i in l]
##
##	print len(last_lines)
##
##	if len(last_lines) > 0:
##		rhos, thetas = zip(*last_lines)
##		print rhos, thetas
##		rho_group = itertools.groupby(rhos)
##		print rho_group.next()[0]
##		theta_group = itertools.groupby(thetas)
##		print theta_group.next()[0]
##
##	return None

def do_green_line_detection(img, framenum=0):

##	cv2.imshow('frame 1', img)

    # Blur img
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
##	cv2.imshow(filename + ' blur', img_blur)

    # Convert to HSV
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

##	# Show HSV histograms
##	draw_hist(img_hsv)

    # Get just green sections
    img_green_bw = cv2.inRange(img_hsv, min_green, max_green)
##	cv2.imshow('green bw', img_green_bw)

    # Close binary image
    img_green_bw = cv2.morphologyEx(img_green_bw, cv2.MORPH_CLOSE, kernel)
##	cv2.imshow('green & closed', img_green_bw)

    # Get image with only largest contour
    mask = get_largest_contour_img(img_green_bw)
##	mask = img_green_bw

##	cv2.imwrite('out_frame_{}_contour.png'.format(framenum), mask)

    # Edge detect (Canny)
    img_edges = cv2.Canny(mask, 1000, 1000, apertureSize=5)
##	cv2.imshow('largest contour edges', img_edges)

    # Set border pixels to zero
    img_edges[0, :] = 0
    img_edges[-1, :] = 0
    img_edges[-2, :] = 0
    img_edges[:, 0] = 0
    img_edges[:, -1] = 0
    img_edges[:, -2] = 0

    # Show edges on original image
    vis = img.copy()
    vis /= 2
    vis[img_edges != 0] = (0, 255, 0)
##	cv2.imshow('edges', vis)

    # Hough lines
    lines = cv2.HoughLines(img_edges, 1, math.pi / 45, tolerance)

    out_lines = []

    if lines is not None:

        for (rho, theta) in lines[0]:
            pt1, pt2 = polar_to_rect(rho, theta)

            cv2.line(vis, pt1, pt2, (0, 0, 255), 1)
            out_lines.append((rho, theta))

##		cv2.imshow('vis lines', vis)

## 	cv2.imwrite('out_frame_{}.png'.format(framenum), vis)

    return vis, out_lines


def polar_to_rect(rho, theta):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (Round(x0 + 1000 * (-b)), Round(y0 + 1000 * (a)))
    pt2 = (Round(x0 - 1000 * (-b)), Round(y0 - 1000 * (a)))
    return pt1, pt2


def get_largest_contour_img(img):

    img_for_contours = img.copy()
    contours, hierarchy = cv2.findContours(
        img_for_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##	print len(contours)

    # Bug workaround - see
    # http://stackoverflow.com/questions/13337058/data-type-error-with-drawcontours-unless-i-pickle-unpickle-first
    contours = cPickle.loads(cPickle.dumps(contours))

    # Isolate largest contour
    contour_sizes = [(i, cv2.contourArea(contour))
                     for i, contour in enumerate(contours)]
##	biggest_contour = max(contour_sizes, key=lambda x: x[1])
##	bc = contours[biggest_contour[0]]
##	print contour_sizes
    big_contours = filter(lambda x: x[1] > big_contour_size, contour_sizes)
    bc = [contours[i] for i in zip(*big_contours)[0]]

##	print bc, contour_sizes

    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, bc, 0, 255, -1)

    return mask


def Round(x):
    return int(x)


def draw_hist(img):
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    ax.set_title('H')
    hist_hue = cv2.calcHist([img], [0], None, [180], [0, 180])
    ax.plot(hist_hue)

    ax = fig.add_subplot(3, 1, 2)
    ax.set_title('S')
    hist_hue = cv2.calcHist([img], [1], None, [256], [0, 255])
    ax.plot(hist_hue)

    ax = fig.add_subplot(3, 1, 3)
    ax.set_title('V')
    hist_hue = cv2.calcHist([img], [2], None, [256], [0, 255])
    ax.plot(hist_hue)

    plt.show()


def wait():
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.chdir('G:\Projects\GrassMow\Video')
    try:
        main('video1out.avi')
        wait()
    except Exception as e:
        print e
        wait()
        raise
