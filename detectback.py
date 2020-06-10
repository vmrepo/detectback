
import cv2 as cv
import numpy as np
from time import time

def drawflow(img, flow, step=16):

    h, w = img.shape[:2]

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)

    lines = np.int32(lines + 0.5)

    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    cv.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

def intersection(x1, y1, angle1, x2, y2, angle2):
    #returned:
    #x, y: infinity for parallel
    #f: -1 - both direct from intersection point; 1 - both direct to intersection point; 0 - different directions;

    eps = np.pi / 180

    if abs(angle1 - angle2) < eps:
        return np.inf, np.inf, 1

    if abs(angle1 - angle2 + np.sign(angle2) * np.pi) < eps:
        return np.inf, np.inf, 0

    if abs(np.pi / 2 - abs(angle1)) < eps:

       k2 = np.tan(angle2)
       b2 = y2 - x2 * k2
       x = x1
       y = k2 * x + b2

    elif abs(np.pi / 2 - abs(angle2)) < eps:

       k1 = np.tan(angle1)
       b1 = y1 - x1 * k1
       x = x2
       y = k1 * x + b1

    else:

       k1 = np.tan(angle1)
       b1 = y1 - x1 * k1
       k2 = np.tan(angle2)
       b2 = y2 - x2 * k2
       x = (b2 - b1) / (k1 - k2)
       y = k1 * x + b1

    a1 = np.arctan2(y - y1, x - x1)
    a2 = np.arctan2(y - y2, x - x2)

    if abs(angle1 - a1) < eps and abs(angle2 - a2) < eps:
        f = 1
    elif not abs(angle1 - a1) < eps and not abs(angle2 - a2) < eps:
        f = -1
    else:
        f = 0

    return x, y, f

def detectback(size, flow):
    #tested on size 426x240

    point_count = 100
    threshold_factor = 0.4
    eps_value_fixed = 1
    eps_threshold_near = 0.2
    eps_angle_near = 45 * np.pi / 180
    eps_angle_far = 20 * np.pi / 180

    ret = 0#0 - change back; 1 - fixed back; 2 - zoom-translation back

    #points count for fixed
    fixeds = 0

    #hypothesis fields: x; y; -1 - unzoom, 1 - zoom; list indexes of points (for x - infinity: y as angle; zoom as 1)
    #hypotheses list
    hypotheses = []
    #best hypothesis index
    hypothesis_idx = -1

    points = []

    k = 0 
    while (True):

        #stop condition and return:
        if k == point_count:
            #print(len(hypotheses))
            #print(hypotheses[hypothesis_idx])
            a = [fixeds, len(hypotheses[hypothesis_idx][3]) if hypothesis_idx != -1 else 0]
            #print(a)
            s = list(reversed(sorted(a)))
            #print(s)
            if s[0] / point_count > threshold_factor:
                ret = np.array(a).argmax() + 1
            break
        k += 1

        x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        fx, fy = flow[y, x].T
        angle = np.arctan2(fy, fx)
        value = np.sqrt(fx*fx + fy*fy)

        #checking fixed
        if value < eps_value_fixed:
            fixeds += 1

        #new point index
        idx_new = len(points)

        #flags for points for adding new hypotheses
        use_points = [True for _ in points]

        #search suitable hypotheses
        for i in range(len(hypotheses)):

            #checking direction

            if hypotheses[i][0] != np.inf:
                dx, dy = (hypotheses[i][0] - x) / size[0], (hypotheses[i][1] - y) / size[1]
                r = np.sqrt(dx * dx + dy * dy)
                a = np.arctan2(dy, dx)
                da = eps_angle_near if r < eps_threshold_near else eps_angle_far
            else:
                a = hypotheses[i][1]
                da = eps_angle_far

            if abs(a - angle) < da or abs(a - angle + np.sign(angle) * np.pi) < da:

                #reset adding hypothesis flags
                for j in range(len(hypotheses[i][3])):
                    use_points[j] = False

                hypotheses[i][3].append(idx_new)
                if len(hypotheses[i][3]) > len(hypotheses[hypothesis_idx][3]):
                    hypothesis_idx = i

        #all points that participate in suitable hypotheses should be excluded to add new hypotheses for the intersection of movements
        #add new hypotheses for the intersection of the motion of existing points
        for i in range(len(use_points)):
            if use_points[i]:
                xc, yc, fc = intersection(x, y, angle, points[i][0], points[i][1], points[i][2])
                if fc != 0:
                    hypotheses.append([xc, yc if xc != np.inf else angle, fc, [i, idx_new]])

        points.append([x, y, angle])

    return ret

if __name__ == '__main__':

    cam = './video/demo.avi'

    cap = cv.VideoCapture(cam)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    #fourcc = int(cap.get(cv.CAP_PROP_FOURCC))

    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)

    out = cv.VideoWriter('./video/demo_processing.avi', fourcc, fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    gray_prev = None
    flow = None
    r_prev = None
    r = None
    r_past = None

    names = ['changed', 'fixed', 'zoom-translation']

    t0 = time()
    nframes = 0

    while True:

        success, img = cap.read()

        if not success:
            break

        nframes += 1

        h, w = img.shape[:2]

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if not gray_prev is None:

            flow = cv.calcOpticalFlowFarneback(gray_prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            r = detectback((w, h), flow)
            if r != r_prev:
                r_past = r_prev
                print(names[r])
                print()
            r_prev = r

        gray_prev = gray

        frm = drawflow(gray, flow) if not flow is None else gray
        if not r is None:
            if not r_past is None:
                cv.putText(frm, 'past: ' + names[r_past], (0, h - 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv.putText(frm, 'now: ' + names[r], (0, h - 2), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv.imshow(cam, frm)

        out.write(frm)

        ch = cv.waitKey(delay)
        if ch == 27:#ESC
            break

    t1 = time()

    print('frames:', nframes)
    print('time:', t1 - t0)

    out.release()
    cap.release()

    cv.destroyAllWindows()
