import cv2
import numpy as np

image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

Z = 800


lower_red = np.array([60, 60, 90])
upper_red = np.array([80, 80, 110])

mask_red_1 = cv2.inRange(image1, lower_red, upper_red)
contours, _ = cv2.findContours(mask_red_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 1 or rect[3] < 1:
        continue
    x, y, w, h = rect
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 5)

Ip1x = (x + (x+w))//2
Ip1Y = (y + (y+h))//2

mask_red_2 = cv2.inRange(image2, lower_red, upper_red)
contours, _ = cv2.findContours(mask_red_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 1 or rect[3] < 1:
        continue
    x, y, w, h = rect
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 5)

Ip2x = (x + (x+w))//2
Ip2Y = (y + (y+h))//2

dist = 450

focal = 685

D = round((dist * focal)/abs(Ip1x - Ip2x), 2)

print("Distance: {} mm".format(D))

cv2.imshow("Image1", image1)
cv2.imshow("Image2", image2)
cv2.waitKey(0)