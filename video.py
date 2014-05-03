import os
import cv2
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
bigkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

SHADOW_SIZE_THRESH = 1000

states = {
    0: 'waiting',
    1: 'reversing far',
    2: 'reversing close',
    3: 'dumping'
}


def main(filename):

    video = cv2.VideoCapture(filename)
    fgbg = cv2.BackgroundSubtractorMOG2(100, 150, True)
    fgbg2 = cv2.BackgroundSubtractorMOG2(100, 150, False)

    # Use first frame for background removal
    playing, frame = video.read()
    fgbg.apply(frame)
    fgbg2.apply(frame)

    STATE = 0

    count = 0
    while(1):
        playing, frame = video.read()

        if not playing:
            return

        # blur image
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg2.apply(frame)

        cv2.imshow('frame', fgmask)

        if STATE < 2:

            opened = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('opened frame', opened)
            shadow_mask, shadow_size = get_shadow_contour(opened)

            if shadow_mask is not None:
                # cv2.imshow('shadow mask', shadow_mask)
                # make shadows blue
                frame[shadow_mask == 255] = (255, 0, 0)

                if STATE == 0:
                    if shadow_size > 4000:
                        STATE = 1
                        print states[STATE]

                if STATE == 1:
                    if shadow_size < 2000:
                        STATE = 2
                        print states[STATE]

        cv2.imshow('raw', frame)

        if STATE == 2:

            closed = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, bigkernel)
            cv2.imshow('closed frame', closed)

        if count > 140:
            wait()
        count += 1


def get_shadow_contour(mask):
    # get just grey (127)
    shadow = cv2.inRange(mask.copy(), 126, 128)
    shadow_contour = np.zeros(shadow.shape, np.uint8)

    contours, hierarchy = cv2.findContours(
        shadow,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0

    # Find large contours
    contour_sizes = [
        (cv2.contourArea(contour), contour)
        for contour in contours
    ]

    max_contour = max(contour_sizes, key=lambda x: x[0])
    contour_size, largest_contour = max_contour

    if contour_size > SHADOW_SIZE_THRESH:
        # fill largest contour
        # import ipdb; ipdb.set_trace()
        cv2.drawContours(shadow_contour, [largest_contour], 0, 255, -1)

    return shadow_contour, contour_size


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
