import os
import cv2
import numpy as np

min_yellow = np.array([25, 0, 0])
max_yellow = np.array([70, 100, 256])

min_rocks = np.array([0, 0, 0])
max_rocks = np.array([180, 30, 256])


def main(filename):

    filename = 'bigrock.mp4'

    video = cv2.VideoCapture(filename)
    f, img = video.read()

    cv2.imshow('raw', img)

    img_blur = cv2.GaussianBlur(img, (11, 11), 0)
    cv2.imshow('blur', img_blur)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    img_yellow_bw = cv2.inRange(img_hsv, min_yellow, max_yellow)
    cv2.imshow('yellow range', img_yellow_bw)

    img_rocks_bw = cv2.inRange(img_hsv, min_rocks, max_rocks)
    cv2.imshow('rocks range', img_rocks_bw)

    


def wait():
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.chdir('/Users/alex/Projects/rio_video')
    try:
        main('bigrock.mp4')
        wait()
    except Exception as e:
        print e
        wait()
        raise
