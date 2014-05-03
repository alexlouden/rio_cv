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

    count = 0
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

        get_contour(opened)

        # closed = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, bigkernel)
        # cv2.imshow('closed frame', closed)

        if count > 100:
            wait()
        count += 1


def get_contour(mask):
    # get just grey (127)
    shadow = cv2.inRange(mask.copy(), 126, 128)
    shadow_contour = np.zeros(shadow.shape, np.uint8)

    contours, hierarchy = cv2.findContours(
        shadow,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    print len(contours)

    if not contours:
        return

    # Find large contours
    contour_sizes = [
        (cv2.contourArea(contour), contour)
        for contour in contours
    ]

    max_contour = max(contour_sizes, key=lambda x: x[0])
    contour_size, largest_contour = max_contour

    print contour_size

    if contour_size > 1000:
        # fill largest contour
        # import ipdb; ipdb.set_trace()
        cv2.drawContours(shadow_contour, [largest_contour], 0, 255, -1)

    cv2.imshow('shadow contour', shadow_contour)

    return shadow_contour


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
