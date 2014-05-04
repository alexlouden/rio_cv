import os
import cv2
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
smallkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
bigkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

SHADOW_SIZE_THRESH = 1000
LARGE_ROCK_THRESH = 300
ROCK_ROUNDNESS = 0.6

states = {
    0: 'waiting',
    1: 'reversing far',
    2: 'reversing close',
    3: 'tipping'
}


def main(filename, show=True):

    if not show:
        cv2.imshow = lambda x, y: None

    video = cv2.VideoCapture(filename)

    fgbg = cv2.BackgroundSubtractorMOG2(100, 150, True)
    fgbg2 = cv2.BackgroundSubtractorMOG2(100, 150, False)

    # Use first frame for background removal
    playing, frame = video.read()
    fgbg.apply(frame)
    fgbg2.apply(frame)

    codecid = "avc1"
    codec = cv2.cv.FOURCC(*codecid)
    frame_size = frame.shape[1::-1]
    filename_out = '_out.'.join(filename.split('.', 1))
    video_out = cv2.VideoWriter(filename_out, codec, 10, frame_size)

    if not video_out.isOpened():
        raise RuntimeError('Output file not open')

    STATE = 0
    print states[STATE]

    count = 0
    while True:
        playing, frame = video.read()

        if not playing:
            break

        # blur image
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg2.apply(frame)

        if STATE < 2:
            # Find shadow region

            cv2.imshow('frame', fgmask)

            opened = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('opened frame', opened)
            # get just grey (127)

            shadow_mask, shadow_size = get_large_contour(
                opened, SHADOW_SIZE_THRESH, 126, 128)

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

        if STATE == 2:
            # Isolate truck trap
            # detect large rocks

            # remove small noise (~3)
            opened = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, smallkernel)

            # close to combine rubble in back of truck
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, bigkernel)
            # cv2.imshow('closed frame', closed)

            # get large background area
            large_mask, mask_size = get_large_contour(closed, 100, 254, 255)

            if large_mask is None:
                continue

            # Background green
            large_mask = cv2.morphologyEx(large_mask, cv2.MORPH_DILATE, kernel)
            frame[large_mask != 255] = (0, 255, 0)

            # ignore background
            fgmask2[large_mask != 255] = 0

            cv2.imshow('frame', fgmask2)
            opened = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, smallkernel)
            # cv2.imshow('opened frame', opened)

            contours, hierarchy = cv2.findContours(
                opened,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            large_rocks_mask = np.zeros(opened.shape, np.uint8)

            # Find large contours
            large_rocks = [
                contour for contour in contours
                if cv2.contourArea(contour) > LARGE_ROCK_THRESH
            ]

            # find rectangular objects
            round_rocks = []
            for rock in large_rocks:
                x, y, w, h = cv2.boundingRect(rock)
                ratio = w / h if w < h else h / w
                if ratio > ROCK_ROUNDNESS:
                    round_rocks.append(rock)
                #import ipdb; ipdb.set_trace()

            cv2.drawContours(large_rocks_mask, round_rocks, -1, 255, -1)
            #cv2.drawContours(large_rocks_mask, round_rocks, -1, 190, -1)
            cv2.imshow('large rocks', large_rocks_mask)
            large_rocks_mask = cv2.morphologyEx(large_rocks_mask, cv2.MORPH_CLOSE, kernel)
            # print len(large_rocks)
            frame[large_rocks_mask == 255] = (0, 0, 255)

        cv2.imshow('raw', frame)
        video_out.write(frame)

        if show:
            wait()

        count += 1

    video_out.release()


def get_large_contour(mask, min_size=100, lbounds=0, ubounds=100):
    shadow = cv2.inRange(mask.copy(), lbounds, ubounds)
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

    if contour_size > min_size:
        # fill largest contour
        # import ipdb; ipdb.set_trace()
        cv2.drawContours(shadow_contour, [largest_contour], 0, 255, -1)

    return shadow_contour, contour_size


def wait():
    cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    os.chdir('/Users/alex/Projects/rio_video')
    import sys
    main(sys.argv[1], False)
