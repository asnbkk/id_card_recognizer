import urllib.request
import ssl
import re
import datetime
import cv2
import numpy as np
import pytesseract
from scipy.ndimage import rotate as Rotate
from skimage.metrics import structural_similarity
from get_names import get_names, get_side_lens, order_points
from pdf2image import convert_from_path
# import urllib.request
import time
from urllib import request

def download_file(download_url, filename, file_type): 
    filename = f'{filename}_{file_type}.{file_type}'
    # request.urlretrieve(download_url, filename)
    with request.urlopen(download_url) as d, open(filename, "wb") as opfile:
        data = d.read()
        opfile.write(data)
    
    return filename

def tesseract_find_rotatation(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, 10, 30, 20)
    thresh = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,1)
    k = pytesseract.image_to_osd(thresh, output_type='dict', lang='rus')
    img_rotated = Rotate(img, 360-k['rotate'])
    return img_rotated, k

def get_bounds_for_names(height, width=0):
    # percent = .25
    height_bound_ = 50 
    width_bound_ = 40
    height_bound = height - height_bound_, height + height_bound_
    width_bound = width - width_bound_, width +  width_bound_ 
    return (height_bound, width_bound)

def set_front_names(front_names_list, front_names_dict):
    for i in front_names_list:
        height = i[-1][-1]
        l, h = get_bounds_for_names(111)[0]
        if l < height < h and front_names_dict['surname'] is None:
            front_names_dict['surname'] = i[0]
        l, h = get_bounds_for_names(251)[0]
        if l < height < h and front_names_dict['name'] is None:
            front_names_dict['name'] = i[0]
        l, h = get_bounds_for_names(396)[0]
        if l < height < h and front_names_dict['fathername'] is None:
            front_names_dict['fathername'] = i[0]
        l, h = get_bounds_for_names(525)[0]
        if l < height < h and front_names_dict['date_of_birth'] is None:
            front_names_dict['date_of_birth'] = i[0]
        
    return front_names_dict

def set_back_names(back_names_list, back_names_dict):
    # print(back_names_list)
    for i in back_names_list:
        height = i[-1][-1]
        width = i[-1][0]
        l, h = get_bounds_for_names(90)[0]
        if l < height < h and back_names_dict['place_of_born'] is None:
            # back_names_dict['place_of_born'] = re.sub(pattern, '', i[0])
            back_names_dict['place_of_born'] = i[0]
        l, h = get_bounds_for_names(202)[0]
        if l < height < h and back_names_dict['nation'] is None:
            back_names_dict['nation'] = re.sub(pattern, '', i[0])
            # back_names_dict['nation'] = i[0]
        l, h = get_bounds_for_names(306)[0]
        # change to default value
        if l < height < h and back_names_dict['organ'] is None:
            # back_names_dict['organ'] = re.sub(pattern, '', i[0])
            back_names_dict['organ'] = i[0]

        l, h = get_bounds_for_names(408)[0]
        l_, h_ = get_bounds_for_names(408, 35)[1]
        if l < height < h and l_ < width < h_ and back_names_dict['s_date'] is None:
            back_names_dict['s_date'] = i[0]

        l, h = get_bounds_for_names(412)[0]
        l_, h_ = get_bounds_for_names(412, 375)[1]
        if l < height < h and l_ < width < h_ and back_names_dict['e_date'] is None:
            back_names_dict['e_date'] = i[0]
    return back_names_dict

# path = './data/30.pdf'

def get_data(filename, is_pdf, card_data, front_names_dict, back_names_dict):
    print(filename)
    img_ = cv2.imread(filename)
    img, _ = tesseract_find_rotatation(img_)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_range = (range(5, 15, 5))
    kernel = np.ones((2,2), np.uint8)  

    canny_lower = range(1, 50, 10)
    canny_upper = range(50, 500, 50)

    ONLY_DIGITS_CONF = '-c tessedit_char_whitelist=0123456789'

    for blur in blur_range:
        try:
            for lower in canny_lower:
                for upper in canny_upper:

                    print(blur, lower, upper)
                    
                    img_blur = cv2.bilateralFilter(img_gray, 10, blur, blur)
                    canny = cv2.Canny(img_blur, lower, upper)
                    thresh = cv2.dilate(canny, kernel, iterations=3)

                    contours, hierarchy = cv2.findContours(thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)
                    con = np.zeros_like(img)
                    # top 5 rects
                    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                    con = cv2.drawContours(con, page, -1, (0, 255, 255), 1)

                    # shit check
                    is_solo = False if len(page) == 2 else True
                    # todo: descew

                    img_orig = img.copy()
                    for index, c in enumerate(page):
                        # Approximate the contour.
                        epsilon = .01 * cv2.arcLength(c, True)
                        corners = cv2.approxPolyDP(c, epsilon, True)
                        if len(corners):
                            corners_ = sorted(np.concatenate(corners).tolist())
                            # print(corners_)
                            pts = order_points(corners_)
                            (tl, tr, br, bl) = pts
                            maxWidth, maxHeight = get_side_lens(*pts)

                            # con = cv2.rectangle(img_orig, pts[0], pts[2], (0, 255, 255), 3)  
                            # cv2.imshow('test', con) 
                            # cv2.waitKey(0)
                            
                            destination_corners = [
                                [0, 0], 
                                [maxWidth, 0], 
                                [maxWidth, maxHeight], 
                                [0, maxHeight]]

                            M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(destination_corners))  # type: ignore

                            final = cv2.warpPerspective(
                                img_orig, 
                                M, 
                                (maxWidth, maxHeight), 
                                flags=cv2.INTER_LINEAR)

                            ratio = maxHeight / maxWidth
                            # print(ratio)
                            final = img_orig[
                                pts[0][-1]:pts[-1][-1], 
                                pts[0][0]:pts[2][0]] if is_pdf else final
                            
                            
                            if ratio > 0.57 and ratio < 0.66:
                                resolution = maxHeight * maxWidth
                                SCALE_PERCENT = 230 # percent of original size
                                width = int(final.shape[1] * SCALE_PERCENT / 100)
                                height = int(final.shape[0] * SCALE_PERCENT / 100)
                                dim = (1642, 1028)
                                
                                # resize image
                                resized = cv2.resize(final, dim, interpolation = cv2.INTER_AREA)
                                height, width, _ = resized.shape

                                # cv2.imshow('test', resized)
                                # cv2.waitKey(0)

                                # if prev 100% simmilar as current
                                if len(res_) > 0:
                                    img2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                                    try:
                                        for i in res_:
                                            img1 = cv2.cvtColor(i['img'], cv2.COLOR_BGR2GRAY)    
                                            score, diff = structural_similarity(img1, img2, full=True)
                                            if score == 1.0:
                                                raise StopLookingForThings()
                                    except StopLookingForThings:
                                        continue
                                
                                # getting iin
                                top = int(height - height / 7)
                                bottom = height
                                left = 0
                                right = int(width / 2)
                                res = resized[top:bottom, left:right]

                                # cv2.imshow('test_maunal_model.jpg', res)
                                # cv2.waitKey(0)
                                
                                text = pytesseract.image_to_string(res, config=ONLY_DIGITS_CONF)
                                for t in text.split('\n'):
                                    if len(t) == 12:  
                                        # print(t)
                                        # side = card_data['side']
                                        # if side == 'back':
                                        #     side = 'both'
                                        # elif side == 'both' or side == 'front':
                                        #     side = side
                                        # else: side = 'front'
                                        # card_data['side'] = side
                                        card_data['side'] = 'front'
                                        card_data['iin'] = t
                                        card_data['is_new'] = True
                                        res_.append({'img': resized, **card_data})

                                top = int(height - height / 4)
                                bottom = int(height - height / 7)
                                left = int(width - width / 3)
                                right = width
                                res = resized[top:bottom, left:right]

                                # cv2.imshow('test_maunal_model.jpg', res)
                                # cv2.waitKey(0)
                                
                                text = pytesseract.image_to_string(res, config=ONLY_DIGITS_CONF)
                                for t in text.split('\n'):
                                    # tesseract dump shits missgets as iin >>>
                                    if len(t) == 12 and int(t[:2]) > 44:
                                        # print(t)
                                        card_data['iin'] = t
                                        card_data['side'] = 'front'
                                        card_data['is_new'] = False
                                        res_.append({'img': resized, **card_data})
                                
                                top = 0
                                bottom = int(height / 7)
                                left = int(width - width / 2.5)
                                right = width
                                res = resized[top:bottom, left:right]

                                # cv2.imshow('test_maunal_model.jpg', res)
                                # cv2.waitKey(0)
                                text = pytesseract.image_to_string(res, config=ONLY_DIGITS_CONF)
                                
                                for t in text.split('\n'):
                                    if len(t) == 9:
                                        # print(t)
                                        if card_data['card_number'] is None:
                                            card_data['card_number'] = t
                                        card_data['side'] = 'back'
                                        res_.append({'img': resized, **card_data})

                                if card_data['side'] == 'back':
                                    top = int(height / 8)
                                    bottom = int(height - height / 2.5)
                                    left = int(width / 3.9) if is_solo else int(width / 4)
                                    right = int(width - width / 8)
                                    res = resized[top:bottom, left:right]
                                    try:
                                        back_names = get_names(res, 'back_names', is_pdf)
                                        # print(back_names)
                                        card_data['back_data'] = set_back_names(back_names, back_names_dict)
                                    except Exception as e:
                                        print(e)
                                        continue

                                if card_data['side'] == 'front' and card_data['is_new']:
                                # getting front data
                                    top = int(height / 4)
                                    bottom = int(height - height / 8)
                                    left = int(width / 2.8) if is_solo else int(width / 2.9)
                                    right = int(width - width / 4)
                                    res = resized[top:bottom, left:right]

                                    # cv2.imshow('test_maunal_model.jpg', res)
                                    # cv2.waitKey(0)
                                    
                                    try:
                                        res_names = get_names(res, 'front_names', is_pdf)
                                        # print(res_names)
                                        card_data['front_data'] = set_front_names(res_names, front_names_dict)
                                    except Exception as e:
                                        print(e)
                                        continue
                                elif card_data['side'] == 'front' and card_data['is_new'] is False:
                                    top = int(height / 3)
                                    bottom = int(height - height / 8)
                                    left = int(width / 3.5)
                                    right = int(width - width / 3)
                                    res = resized[top:bottom, left:right]
                                    # cv2.imshow('test', res_)
                                    # cv2.waitKey(0)
                                    try:
                                        res_names = get_names(res, 'front_names', is_pdf)
                                        card_data['front_data'] = set_front_names(res_names, front_names_dict)
                                    except Exception as e:
                                        print(e)
                                        continue
                                
                                # print('=' * 30)
                                # print(front_names_dict)
                                # print(back_names_dict)
                                # print('=' * 30)
                                
                                # print(card_data)
                                # obj_res.append(card_data)
                                
                                if is_solo:
                                    if card_data['side'] == 'front':
                                        if card_data.get('iin') is not None and all(front_names_dict.values()):
                                            raise StopLoop()
                                        else:
                                            raise CannotGetData()
                                    if card_data['side'] == 'back':
                                        if card_data.get('card_number') is not None and all(back_names_dict.values()):
                                            raise StopLoop()
                                        else:
                                            raise CannotGetData()
                                else:
                                    if all(front_names_dict.values()) and all(back_names_dict.values()):
                                        raise StopLoop()
        except StopLoop:
            break
        except CannotGetData:
            return 'Can not read id'

    return card_data


# print(card_data)

today = datetime.date.today()
year = today.year % 100

class StopLookingForThings(Exception): pass
class StopLoop(Exception): pass
class CannotGetData(Exception): pass

is_solo, is_pdf = None, None
pattern = '[^а-яА-Я]+' # for russian names
res_, obj_res = [], []



# card_data_ = card_data.copy()
# back_names_dict_ = back_names_dict.copy()
# front_names_dict_ = front_names_dict.copy()

def main(link, filetype):
    card_data = {
    'side': None,
    'iin': None,
    'card_number': None,
    'back_data': {},
    'front_data': {}
    }

    back_names_dict = {
        'place_of_born': None,
        'nation': None,
        'organ': None,
        's_date': None,
        'e_date': None
    }

    front_names_dict = {
        'surname': None,
        'name': None,
        'fathername': None,
        'date_of_birth': None
    }
    # ssl._create_default_https_context = ssl._create_unverified_context
    path = download_file(link, './temp_data/temp', filetype)    
    start_time = time.time()
    # path_ = path.split('.')
    # file_extension = path_[-1]
    print(filetype, link)

    if filetype == 'pdf':
        pages = convert_from_path(path)
        for page in pages:
            page.save(path, 'JPEG')
        is_pdf = True
    elif filetype == 'jpg':
        is_pdf = False
    
    output = get_data(path, is_pdf, card_data, front_names_dict, back_names_dict)
    print("--- %s seconds ---" % (time.time() - start_time))

    # global card_data, front_names_dict, back_names_dict
    # card_data = card_data_
    # front_names_dict = front_names_dict_
    # back_names_dict = back_names_dict_
    
    return output


# main(link, filetype)

# print(len(res_))

# for r in res_:
#     cv2.namedWindow('img_file_name', cv2.WINDOW_NORMAL) # Creates a window
#     os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
#     cv2.imshow('test', r['img'])
#     cv2.waitKey(0)