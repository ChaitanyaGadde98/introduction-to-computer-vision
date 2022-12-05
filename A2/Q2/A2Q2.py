import cv2

test_images = {'test1': ['files/Art and humanities01_team6.jpg', 'files/Art and humanities02_team6.jpg',
                         'files/Art and humanities03_team6.jpg'],
               'test2': ['files/bookstore1.jpg', 'files/bookstore2.jpg', 'files/bookstore3.jpg'],
               'test3': ['files/classroom_south1.jpg', 'files/classroom_south2.jpg', 'files/classroom_south3.jpg'],
               'test4': ['files/dahlberg1.jpg', 'files/dahlberg2.jpg', 'files/dahlberg3.jpg'],
               'test5': ['files/sportsarena1.jpg', 'files/sportsarena2.jpg', 'files/sportsarena3.jpg']}

# initialized a list of images
for key, value in test_images.items():
    images = []
    for i in range(len(value)):
        images.append(cv2.imread(value[i]))
        images[i] = cv2.resize(images[i], (0, 0), fx=0.4, fy=0.4)
    image_sticher = cv2.Stitcher.create()
    (dummy, output) = image_sticher.stitch(images)

    cv2.imshow('Stitched Image', output)
    cv2.imwrite(key+'.jpg', output)
    cv2.waitKey(0)
