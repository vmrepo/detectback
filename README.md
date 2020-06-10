
Detection the movements of the background on the video.

Background moves can be divided down into a full background change or background offset in combination with zoom in or zoom out.
Analysis of the background motion may be useful for detection the structure of video and tracking objects.
The algorithm is based on the detection of the regularity of pixel motion. The motion of pixels is obtained by detecting the optical flow.

Assumptions that are used in the algorithm.
Change of background - pixels do not match, change the maximum random.
Strong distortions - it is impossible to distinguish the regularity of the pixels, local regularities for individual moving objects are not rebuilt.
Change of background impossible to distinguish from a strong distortions here include rotation.
Possible to detection the absence of motion, offset and zoom.

Optical flow is a set of motion vectors of image pixels from one frame to another.
The displacement background vectors are oriented in one direction.
Zoom in or zoom out - vector converge at the center point, if not the center, then there is an offset.

The idea of the algorithm.
A list of hypotheses is constructed: randomly iterate over the points of the frame (you can do everything, but it is expensive), the point is checked and refers to any hypothesis, new hypotheses are added.
The most mass hypothesis is detected or it is concluded that there is no such hypothesis (how quickly to stop is an algorithm parameter).

preparedemo.py
Helper, from the classic vtest.avi is preparing a demo.avi with different motions of the background.
Record demo.avi stops when you exit by pressing escape.

detectback.py
Analyzes the movement background for the demo.avi and saves demo demo_processing.avi.
