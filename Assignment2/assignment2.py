
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import tqdm


class image():
    def __init__(self, img):
        self.img_bgr = img
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255
        # self.gray = cv.GaussianBlur(self.gray,(3,3),1)
        self.ix = self.Ix()
        self.iy = self.Iy()
        self.gradient_angel = np.arctan2(self.iy, self.ix) *180. / np.pi
        self.gradient_angel[self.gradient_angel < 0] += 360

    # calculating gradient in x direction
    def Ix(self):
        sobel_x = (1/8)*np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        return cv.filter2D(self.gray, -1, kernel=sobel_x, borderType=cv.BORDER_CONSTANT)

    # calculating gradient in y direction
    def Iy(self):
        sobel_y = (1/8)*np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return cv.filter2D(self.gray, -1, kernel=sobel_y, borderType=cv.BORDER_CONSTANT)

    # def quantize_angel(self):
    #     bins = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    #     digitized = np.digitize(self.gradient_angel, bins)
    #     digitized[digitized == 0] = 8
    #     return digitized

    # map angels to bins
    def quantize_angel(self):
        digitized = self.gradient_angel // 45
        digitized[digitized == 8] = 0
        return digitized.astype(int)


class keypoints():
    def __init__(self, img, harris_thresh, nonmax='default', adaptive_point_count=200):
        self.img = img
        self.harris_response = self.Harris(harris_thresh)
        if nonmax == 'default':
            self.keypoints = self.nonmax_suppression()
        elif nonmax == 'adaptive':
            self.keypoints, self.adaptive_radious = self.adaptive_non_max(adaptive_point_count)

        self.kps = [cv.KeyPoint(point[1], point[0], 1) for point in self.keypoints]

    # calculate harris thresholded response for all points
    def Harris(self, thresh, alpha=0.04):

        ixx = cv.GaussianBlur(self.img.ix ** 2, (5, 5), sigmaX=1, sigmaY=1, borderType=cv.BORDER_CONSTANT)
        iyy = cv.GaussianBlur(self.img.iy ** 2, (5, 5), sigmaX=1, sigmaY=1, borderType=cv.BORDER_CONSTANT)
        ixy = cv.GaussianBlur(self.img.ix * img.iy, (5, 5), sigmaX=1, sigmaY=1, borderType=cv.BORDER_CONSTANT)

        sum_filter = np.ones((5, 5))

        sxx = cv.filter2D(ixx, -1, sum_filter, borderType=cv.BORDER_CONSTANT)
        syy = cv.filter2D(iyy, -1, sum_filter, borderType=cv.BORDER_CONSTANT)
        sxy = cv.filter2D(ixy, -1, sum_filter, borderType=cv.BORDER_CONSTANT)

        det = (sxx * syy) - (sxy ** 2)
        trace = sxx + syy
        response = det - alpha * (trace ** 2)

        thresholded_response = (response > thresh).astype(int) * response
        return thresholded_response

    def nonmax_suppression(self):
        w, h = self.harris_response.shape
        mat = np.copy(self.harris_response)
        for i in tqdm.tqdm(range(w)):
            for j in range(h):
                try:
                    patch = np.zeros((3, 3))
                    a = mat[i - 1:i + 2, j - 1:j + 2]
                    argmax = np.unravel_index(np.argmax(a, axis=None), a.shape)
                    patch[argmax] = a[argmax]
                    mat[i - 1:i + 2, j - 1:j + 2] = patch
                except:
                    pass
        return np.argwhere(mat > 0)

    def show_keypoints(self):
        return cv.drawKeypoints(self.img.img_bgr, self.kps, None)

    def adaptive_non_max(self, points_count):
        points = np.argwhere(self.harris_response > 0)
        n, _ = points.shape
        radious = np.zeros(n)
        radious += 10000000
        for i in tqdm.tqdm(range(n)):
            for j in range(n):
                if self.harris_response[points[i][0], points[i][1]] < self.harris_response[points[j][0], points[j][1]] and i != j:
                    distance = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                    if distance < radious[i]:
                        radious[i] = distance
        sorted_radious = np.argsort(radious)
        return points[sorted_radious[-points_count:]], np.sort(radious)[-points_count:]

    # SIFT keypoint descriptor
    def descriptor(self):
        describtors = []
        new_points = []
        w, h = self.img.gray.shape

        for point in self.keypoints:
            if point[0] - 8 < 0 or point[0] + 8 > w or point[1] - 8 < 0 or point[1] + 8 > h:
                continue
            # print(self.img.gray.shape)
            # patch = self.img.gray[point[0] - 8:point[0] + 8, point[1] - 8:point[1] + 8]
            digitized = self.img.quantize_angel()[point[0] - 8:point[0] + 8, point[1] - 8:point[1] + 8]
            describtor = np.zeros((16, 8))
            _, patch_counts = np.unique(digitized, return_counts=True)
            dominant_gradient = np.argmax(patch_counts)
            for i in range(4):
                for j in range(4):
                    unique, counts = np.unique(digitized[i * 4:(i * 4) + 4, j * 4:(j * 4) + 4], return_counts=True)
                    cell_bins = np.zeros(8)
                    for k, value in enumerate(unique):
                        cell_bins[value - 1] = counts[k]

                    # normalizing
                    cell_bins /= 16
                    cell_bins = np.roll(cell_bins, -dominant_gradient)
                    # illumination invariance
                    cell_bins[cell_bins > .2] = .2
                    cell_bins = cell_bins ** 2 / np.sum(cell_bins ** 2)
                    # rotation alignment

                    describtor[i * 4 + j, :] = cell_bins
            describtor.ravel()
            describtors.append(describtor.ravel())

        return np.stack(describtors, axis=0)


def match(descriptor1, descriptor2, mode='ratio_test', threshold=0.4):

    n, _ = descriptor1.shape
    m, _ = descriptor2.shape

    distance = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distance[i, j] = np.sum((descriptor1[i, :] - descriptor2[j, :]) ** 2)
    sort = np.argsort(distance, axis=1)
    print(f'min:{np.min(distance)}')
    if mode == 'threshold':
        matched = []
        for i in range(n):
            if distance[i, sort[i, 0]] < threshold:
                matched.append([i, sort[i, 0]])
        return np.stack(matched)
    elif mode == 'ratio_test':
        matched = []
        for i in range(n):
            ratio = distance[i, sort[i, 0]] / distance[i, sort[i, 1]]
            if ratio < threshold:
                matched.append([i, np.argmin(distance[i, :])])
        return np.stack(matched)
    else:
        raise ('method is not implemented')


path = './image_sets/yosemite/'

imgs = []
features = []
for name in os.listdir(path):
    print(os.path.join(path, name))
    img = image(cv.imread(os.path.join(path, name)))
    feature = keypoints(img, harris_thresh=0.01, nonmax='default', adaptive_point_count=200)
    # feature.descriptor()
    cv.imshow('kps', feature.show_keypoints())
    cv.waitKey(0)
    print(f'keypoint:{feature.keypoints.shape[0]}')
    features.append(feature)
    imgs.append(img)

desc1 = features[0].descriptor()
desc2 = features[2].descriptor()

cv.imshow('kp1',features[0].show_keypoints())
cv.imshow('kp2',features[1].show_keypoints())
cv.waitKey(0)
matched = match(desc1, desc2, threshold=.8, mode='ratio_test')
print(f'matched:{matched.shape[0]}')
dmatch = [cv.DMatch(matched[i, 0], matched[i, 1], 0) for i in range(matched.shape[0])]
output = cv.drawMatches(imgs[0].img_bgr, features[0].kps, imgs[1].img_bgr, features[1].kps, dmatch, None)
cv.imwrite('match.png', output)

# cv.imshow('match',output)
# cv.waitKey(0)
