import numpy as np
import cv2


def get_homography(frame1, frame2, good_points):
    query_pts = np.float32([kp_img1[m.queryIdx]
                           .pt for m in good_points]).reshape(-1, 1, 2)
    train_pts = np.float32([kp_img2[m.trainIdx]
                           .pt for m in good_points]).reshape(-1, 1, 2)
    hm_matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    h, w, _ = frame1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, hm_matrix)
    homography = cv2.polylines(frame2, [np.int32(dst)], True, (255, 0, 0), 3)
    return homography, hm_matrix


def extract_features(frame1, frame2):
    sift = cv2.SIFT_create()
    kp_img1, desc_img1 = sift.detectAndCompute(frame1, None)
    kp_img2, desc_img2 = sift.detectAndCompute(frame2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_img1, desc_img2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)
    return good_points, kp_img1, kp_img2


def draw_matches(frame1, frame2, kp_img1, kp_img2, good_points):
    image = cv2.drawMatches(frame1, kp_img1, frame2, kp_img2, good_points, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return image


def visualize_image(frame_name, frame):
    cv2.imshow(frame_name, frame)
    cv2.imwrite(frame_name+".jpg", frame)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture("files/video.mp4")
    frames_dict = {}
    count = 0
    while cap.read()[0]:
        frames_dict[count] = cap.read()[1]
        count += 1

    frame1 = frames_dict[110]
    visualize_image("Frame1", frame1)

    frame2 = frames_dict[120]
    visualize_image("Frame2", frame2)

    good_points, kp_img1, kp_img2 = extract_features(frame1, frame2)
    homography, hm_matrix = get_homography(frame1, frame2, good_points)
    image = draw_matches(frame1, frame2, kp_img1, kp_img2, good_points)

    visualize_image("Homographed Image", homography)
    visualize_image("Feature Matching", image)
