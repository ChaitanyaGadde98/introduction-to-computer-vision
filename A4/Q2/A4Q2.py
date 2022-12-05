import cv2
import depthai as dai
import numpy as np
import webbrowser

DIM = (720, 480)

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

qrDecoder = cv2.QRCodeDetector()

load_image = False


def open_url(url):
    if data != '':
        webbrowser.open(url)
        return True


if load_image:
    image = cv2.imread('vCard.JPG')
    data, bbox, rectifiedImage = qrDecoder.detectAndDecode(image)
    if len(data) > 0:
        print("Decoded Data : {}".format(data))
        rectifiedImage = np.uint8(rectifiedImage);
        cv2.imshow("Rectified QRCode", rectifiedImage);

    open_url(data)
    if cv2.waitKey(1) == ord('q'):
        pass
else:

    with dai.Device(pipeline) as device:
        # Output queue will be used to get the disparity frames from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        while True:

            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            frame = cv2.resize(frame, DIM, interpolation=cv2.INTER_AREA)

            # Detect and decode the qrcode
            data, bbox, rectifiedImage = qrDecoder.detectAndDecode(frame)
            if len(data) > 0:
                print("Decoded Data : {}".format(data))
                qrectifiedImage = np.uint8(rectifiedImage);
                cv2.imshow("Rectified QRCode", rectifiedImage);

            cv2.imshow("QR", frame)

            if cv2.waitKey(1) == ord('q') or open_url(data):
                break
