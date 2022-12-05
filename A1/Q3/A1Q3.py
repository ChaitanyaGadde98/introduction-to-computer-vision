import cv2
import depthai as dai
import numpy as np
import time

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# camera configurations
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
depth = pipeline.createStereoDepth()
left.out.link(depth.left)
right.out.link(depth.right)
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

prev_frame_time = 0
new_frame_time = 0
DIM = (720, 480)

with dai.Device(pipeline) as device:

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:

        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

        inRgb = qRgb.get()

        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()

        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        frame_rgb = cv2.resize(inRgb.getCvFrame(), DIM, interpolation=cv2.INTER_AREA)

        cv2.imshow("rgb", frame_rgb)
        cv2.imshow("disparity", frame)

        print("FPS: ", fps)
        if cv2.waitKey(1) == ord('q'):
            break