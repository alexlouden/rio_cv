import os
import cv2
import numpy as np

min_yellow = np.array([25, 0, 0])
max_yellow = np.array([70, 100, 256])

min_rocks = np.array([0, 0, 0])
max_rocks = np.array([180, 30, 256])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
bigkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))


def main(filename):

    video = cv2.VideoCapture(filename)
    fgbg = cv2.BackgroundSubtractorMOG2(100, 150, True)
    fgbg2 = cv2.BackgroundSubtractorMOG2(100, 150, False)

    while(1):
        playing, frame = video.read()

        if not playing:
            return

        # blur image
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg2.apply(frame)

        cv2.imshow('raw', frame)

        cv2.imshow('frame', fgmask)

        opened = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('opened frame', opened)

        closed = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, bigkernel)
        cv2.imshow('closed frame', closed)

        wait()


def wait():
    cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    os.chdir('/Users/alex/Projects/rio_video')
    # try:
    main('background.mp4')
    wait()
    # except Exception as e:
    #     print e
        # wait()
        # raise
