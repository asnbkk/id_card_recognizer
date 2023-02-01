import numpy as np
import cv2
import pytesseract

class StopLoop(Exception): pass

padding_ver = 12

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect.astype('int').tolist()

def get_side_lens(tl, tr, br, bl):
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight

def get_tess_conf(img, conf_level):
    output = pytesseract.image_to_data(img,
                                       lang='kaz+rus',
                                       nice=-1,
                                       output_type=pytesseract.Output.DICT,
                                       config='--psm 1 tessedit_char_whitelist .123456789')
    conf_list_ = [i for i in output['conf'] if i != -1]
    conf = sum(conf_list_) / len(conf_list_)

    text = ' '.join([i for i in output['text'] if i != ''])
    print(conf, text)

    return text if conf >= conf_level else None

def get_names(img, side='front_names', is_pdf=False):
    res_list = []
    amount = 4 if side == 'front_names' else 5
    
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # deskewed = deskew(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1,8), np.uint8)  
    img_erosion = cv2.erode(img_gray, kernel, iterations=7)

    img_blur = cv2.bilateralFilter(img_erosion, 10, 30, 20)
    kernel = np.ones((10, 30),np.uint8)
    img_blur = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel, iterations=3)

    canny_lower = range(1, 50, 10)
    canny_upper = range(50, 500, 50)
    # for lower in canny_lower:
        # try:
            # for upper in canny_upper:
    kernel = np.ones((2,2), np.uint8)  
    canny = cv2.Canny(img_blur, 1, 50)
    thresh = cv2.dilate(canny, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('test', thresh)
    # cv2.waitKey(0)
    
    # con = np.zeros_like(img)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:amount]

    pts_list = []
    for c in page:
        # Approximate the contour
        epsilon = .01 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # if len(corners):
        corners_ = sorted(np.concatenate(corners).tolist())
        pts = order_points(corners_)
        # if maxHeight > 20:
        pts_list.append(pts)
        # con = cv2.rectangle(img_blur, pts[0], pts[2], (0, 255, 255), 3)  
        # cv2.imshow('test', con) 
        # cv2.waitKey(0)

    # if len(pts_list) == amount - 1:
        # raise StopLoop()
# except StopLoop:
# break
            
    # up_to_down_pts = sorted(pts_list, key=(lambda x: x[0][-1]))

    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    
    for pts in pts_list:
        padding_top = padding_ver + 5 if is_pdf else padding_ver + 7
        name = img[
            pts[0][-1]-padding_top:pts[-1][-1]+padding_ver + 5, 
            pts[0][0]-5:pts[2][0]-5]
        try:
            # norm_img = np.zeros((img.shape[0], img.shape[1]))
            # img = cv2.normalize(name, norm_img, 0, 255, cv2.NORM_MINMAX)

            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
            # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            res = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
            res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            scale_percent = 150 # percent of original size
            width = int(res.shape[1] * scale_percent / 100)
            height = int(res.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(res, dim, interpolation = cv2.INTER_AREA)

            # cv2.imshow('test', resized)
            # cv2.waitKey(0)
                
            res_ = get_tess_conf(resized, 40)
            if res_ != '':
                res_list.append((res_, pts[0]))
        except Exception as e:
            print(e)
            continue
    return res_list

# img = cv2.imread('./data/cropped2.jpg')
# res = get_names(img)
# print(res)



