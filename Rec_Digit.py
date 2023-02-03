import cv2
import numpy as np
from Seg7 import Segments


def scale_up(img):
    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


# load image
img = cv2.imread("segzoom.png")
cv2.imshow("orig0", img)
cv2.waitKey(0)

img = scale_up(img)



# crop
#img = img[300:900,100:800,:]

# lab
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(lab)

# show
cv2.imshow("orig", img)
cv2.waitKey(0)

# closing operation
kernel = np.ones((5,5), np.uint8)

# threshold params
low = 165
high = 200
iters = 3

# make copy
copy = b.copy()

# threshold
thresh = cv2.inRange(copy, low, high)
cv2.imshow("k", thresh)
cv2.waitKey(0)

# dilate
for a in range(iters):
    thresh = cv2.dilate(thresh, kernel)
    cv2.imshow("k1", thresh)
    cv2.waitKey(0)

# erode
for a in range(iters):
    thresh = cv2.erode(thresh, kernel)
    cv2.imshow("k2", thresh)
    cv2.waitKey(0)

# show image
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    cv2.drawContours(img, [contour], 0, (0,255,0), 3)

bounds = []
h, w = img.shape[:2]
for contour in contours:
    left = w
    right = 0
    top = h
    bottom = 0
    for point in contour:
        point = point[0]
        x, y = point
        if x < left:
            left = x
        if x > right:
            right = x
        if y < top:
            top = y
        if y > bottom:
            bottom = y
    tl = [left, top]
    br = [right, bottom]
    bounds.append([tl, br])



# crop out each number
cuts = []
number = 0
for bound in bounds:
    tl, br = bound
    cut_img = thresh[tl[1]:br[1], tl[0]:br[0]]
    cuts.append(cut_img)
    number += 1
    cv2.imshow(str(number), cut_img)
    cv2.waitKey(0)

font = cv2.FONT_HERSHEY_SIMPLEX
model = Segments()
index = 0
for x in range(len(cuts)):
    # save image
    #cv2.imwrite(str(index) + "_" + str(number) + ".jpg", cut)

    # process
    model.digest(cuts[x])
    number = model.getNum() #THIS IS THING WE PASS TO OTHER PROGRAM 

    #showing values guessed by the system 
    print(number)
    cv2.imshow(str(index), cuts[x])
    cv2.waitKey(0)

    '''
    # draw and save again
    h, w = cut.shape[:2]
    drawn = np.zeros((h, w, 3), np.uint8)
    drawn[:, :, 0] = cut
    drawn = cv2.putText(drawn, str(number), (10,30), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imwrite("drawn" + str(index) + "_" + str(number) + ".jpg", drawn)
    
    index += 1
    cv2.waitKey(0)
    '''


