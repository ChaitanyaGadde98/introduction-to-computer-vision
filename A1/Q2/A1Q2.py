import numpy as np
import cv2
import math

FX = 685.61
FY = 685.66
Z = 900


def convert_milli_to_inch(x):
    x = x / 10
    return x / 25.4


image = cv2.imread("image.jpg")

lower_yellow = np.array([110, 215, 210])
upper_yellow = np.array([140, 240, 240])

mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100:
        continue
    x, y, w, h = rect

    Ip1x = x + w
    Ip1y = y
    Ip2x = x + w
    Ip2y = y + h

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.line(image, (Ip2x, Ip1y), (Ip2x, Ip2y), (0, 0, 255), 8)

    Rp1x = Z * (Ip1x / FX)
    Rp1y = Z * (Ip1y / FY)
    Rp2x = Z * (Ip2x / FX)
    Rp2y = Z * (Ip2y / FY)

    dist = math.sqrt((Rp2y - Rp1y) ** 2 + (Rp2x - Rp1x) ** 2)
    val = round(convert_milli_to_inch(dist), 2)
    print("Length of a yellow square: {}".format(val))
    cv2.putText(image, str(val) + " inches", (Ip2x + 20, (y + y + h) // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

lower_blue = np.array([40, 30, 20])
upper_blue = np.array([115, 60, 100])

mask_blue = cv2.inRange(image, lower_blue, upper_blue)
contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100:
        continue
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

    Ip1x = x + w
    Ip1y = y
    Ip2x = x + w
    Ip2y = y + h

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.line(image, (Ip1x, Ip1y), (Ip1x, Ip2y), (0, 0, 255), 8)

    Rp1x = Z * (Ip1x / FX)
    Rp1y = Z * (Ip1y / FY)
    Rp2x = Z * (Ip2x / FX)
    Rp2y = Z * (Ip2y / FY)

    dist = math.sqrt((Rp2y - Rp1y) ** 2 + (Rp2x - Rp1x) ** 2)
    val = round(convert_milli_to_inch(dist), 2)
    print("Diameter of blue cirlce: {}".format(val))
    cv2.putText(image, str(val) + " inches", (Ip1x + 50, (y + y + h) // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
