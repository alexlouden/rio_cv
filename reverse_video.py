import cv2


def reverse(filename):
    video_in = cv2.VideoCapture(filename)

    prop_codec = cv2.cv.CV_CAP_PROP_FOURCC
    prop_fps = cv2.cv.CV_CAP_PROP_FPS
    prop_height = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
    prop_width = cv2.cv.CV_CAP_PROP_FRAME_WIDTH

    codec = int(video_in.get(prop_codec))
    fps = int(video_in.get(prop_fps))
    height = int(video_in.get(prop_height))
    width = int(video_in.get(prop_width))

    size = width, height

    filename_out = '_reversed.'.join(filename.split('.', 1))
    video_out = cv2.VideoWriter(filename_out, codec, fps, size)

    frames = []
    while True:
        playing, frame = video_in.read()
        if not playing:
            break

        frames.append(frame)

    frames.reverse()
    for frame in frames:
        video_out.write(frame)

    video_out.release()


if __name__ == '__main__':
    import sys
    reverse(sys.argv[1])
