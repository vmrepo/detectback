
import cv2 as cv
import numpy as np

def transform(fps, count, shift, scale, img):

    def nothing():
        return

    d = int(1000 / fps)
    t = d * count

    t1, t2, t3, t4, t5, t6, t7 = 2000, 4000, 5000, 7000, 8000, 9000, 11000

    if t < t1:
        #(, t1)
        nothing()

    elif t < t2:
        #(t1, t2)
        nothing()

        scale_start = 1
        scale_target = 3
        scale += (scale_target - scale_start) / ((t2 - t1) / d)
        #print(scale)

    elif t < t3:
        #(t2, t3)
        nothing()

    elif t < t4:
        #(t3, t4)
        nothing()

        shift_start = 0, 0
        shift_target = -100, -100
        shift = shift[0] + (shift_target[0] - shift_start[0]) / ((t4 - t3) / d), shift[1] + (shift_target[1] - shift_start[1]) / ((t4 - t3) / d)
        #print(shift)

    elif t < t5:
        #(t4, t5)
        nothing()

    elif t < t6:
        #(t5, t6)
        nothing()

        scale = 3
        shift = 100, -100

    elif t < t7:
        #(t6, t7)
        nothing()

        scale_start = 3
        scale_target = 1
        scale += (scale_target - scale_start) / ((t7 - t6) / d)

        shift_start = 100, -100
        shift_target = 0, 0
        shift = shift[0] + (shift_target[0] - shift_start[0]) / ((t7 - t6) / d), shift[1] + (shift_target[1] - shift_start[1]) / ((t7 - t6) / d)

        #print(scale, shift)

    else:
        #(t7, )
        nothing()
  
    h, w = img.shape[:2]
    scale = 1 if scale < 1 else scale
    h_, w_ = int(scale * h), int(scale * w)
    sx, sy = int((w_ - w) / 2 + shift[0] * scale), int((h_ - h) / 2 + shift[1] * scale)
    img = cv.resize(img, (w_, h_))
    img = img[sy:sy+h, sx:sx+w]

    return shift, scale, img

if __name__ == '__main__':

    cam = './video/vtest.avi'

    cap = cv.VideoCapture(cam)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    #fourcc = int(cap.get(cv.CAP_PROP_FOURCC))

    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)

    out = cv.VideoWriter('./video/demo.avi', fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    scale = 1
    shift = (0, 0)
    count = 0

    while True:

        success, img = cap.read()

        if not success:
            break

        shift, scale, img = transform(fps, count, shift, scale, img)

        cv.imshow(cam, img)
        out.write(img)

        ch = cv.waitKey(delay)

        if ch == 27:#ESC
            break

        count += 1

    out.release()
    cap.release()

    cv.destroyAllWindows()
