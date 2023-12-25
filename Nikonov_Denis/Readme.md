from google.colab import files
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def align_images(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = sorted(matches, key=lambda x: x.distance)

    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("/content/matches.png", im_matches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(im1, h, (width, height))

    return im1_reg

# Загрузка изображений из Colab
uploaded = files.upload()

# Первый файл - это ссылка, второй - изображение для выравнивания
ref_filename = list(uploaded.keys())[0]
im_filename = list(uploaded.keys())[1]

# Чтение изображений
im_reference = cv2.imread(ref_filename, cv2.IMREAD_COLOR)
im = cv2.imread(im_filename, cv2.IMREAD_COLOR)

# Выравнивание изображений
im_reg = align_images(im, im_reference)

# Сохранение выровненного изображения
out_filename = "/content/Gzuz187.png"
cv2.imwrite(out_filename, im_reg)

print("Сохраненное выровненное изображение: ", out_filename)
