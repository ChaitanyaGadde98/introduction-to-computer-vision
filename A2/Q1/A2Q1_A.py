import numpy as np
from scipy import ndimage
import cv2


def canny_edge_detector(image):
    threshold_weak = None
    threshold_strong = None
    # conversion of image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction step
    sigma = 5
    kernel_size = 5
    size = int(kernel_size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    gaussian_kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal

    image = cv2.filter2D(src=image, kernel=gaussian_kernel, ddepth=19)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(image, Kx)
    Iy = ndimage.filters.convolve(image, Ky)

    G = np.hypot(Ix, Iy)
    magnitude = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    # setting the minimum and maximum thresholds
    # for double thresholding
    max_magnitude = np.max(magnitude)
    if not threshold_weak:
        threshold_weak = max_magnitude * 0.1

    if not threshold_strong:
        threshold_strong = max_magnitude * 0.5

    # getting the dimensions of the input image
    height, width = image.shape

    # Looping through every pixel of the grayscale
    # image
    for x in range(width):
        for y in range(height):

            grad_ang = theta[y, x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang <= 22.5:
                n1x, n1y = x - 1, y
                n2x, n2y = x + 1, y

            # top right (diagonal-1) direction
            elif 22.5 < grad_ang <= (22.5 + 45):
                n1x, n1y = x - 1, y - 1
                n2x, n2y = x + 1, y + 1

            # In y-axis direction
            elif (22.5 + 45) < grad_ang <= (22.5 + 90):
                n1x, n1y = x, y - 1
                n2x, n2y = x, y + 1

            # top left (diagonal-2) direction
            elif (22.5 + 90) < grad_ang <= (22.5 + 135):
                n1x, n1y = x - 1, y + 1
                n2x, n2y = x + 1, y - 1

            # Now it restarts the cycle
            elif (22.5 + 135) < grad_ang <= (22.5 + 180):
                n1x, n1y = x - 1, y
                n2x, n2y = x + 1, y

            # Non-maximum suppression step
            if width > n1x >= 0 and height > n1y >= 0:
                if magnitude[y, x] < magnitude[n1y, n1x]:
                    magnitude[y, x] = 0
                    continue

            if width > n2x >= 0 and height > n2y >= 0:
                if magnitude[y, x] < magnitude[n2y, n2x]:
                    magnitude[y, x] = 0

    ids = np.zeros_like(image)
    # double thresholding step
    for x in range(width):
        for y in range(height):

            grad_mag = magnitude[y, x]

            if grad_mag < threshold_weak:
                magnitude[y, x] = 0
            elif threshold_strong > grad_mag >= threshold_weak:
                ids[y, x] = 1
            else:
                ids[y, x] = 2

    # finally returning the magnitude of
    # gradients of edges
    return magnitude


def harris_corners_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image


def visualize_image(frame_name, frame):
    cv2.imshow(frame_name, frame)
    cv2.imwrite(frame_name + ".jpg", frame)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture("files/video.mp4")
    frames_dict = {}
    count = 0
    while cap.read()[0]:
        frames_dict[count] = cap.read()[1]
        count += 1

    frame = frames_dict[120]
    cv2.imshow("Frame", frame)
    visualize_image("Frame", frame)

    canny_image = canny_edge_detector(frame)
    visualize_image("Canny", canny_image)

    harris_image = canny_edge_detector(frame)
    visualize_image("Harris", harris_image)
