import cv2 as cv
import numpy as np
from copy import deepcopy 
import glob 
from tqdm import tqdm
from image_paths import *

COLORS = ["BLUE", "ORANGE", "YELLOW", "RED", "GREEN", "WHITE"]
COLOR_CODE_MAP = {
    "RED": "R",
    "BLUE": "B",
    "GREEN": "G",
    "YELLOW": "Y",
    "ORANGE": "O",
    "WHITE": "W"
}
COLOR_FILTERS = [
                    (np.array([83,156,121]), np.array([129, 255,255])), #blue
                    (np.array([3,140, 86]), np.array([9, 255, 255])), #orange
                    (np.array([25,145, 143]), np.array([90, 255,255])), #yellow
                    (np.array([136,119,97]), np.array([180, 255,255])), #red
                    (np.array([49,160,64]), np.array([104, 255, 127])), #green
                    (np.array([56,0,146]), np.array([150, 146,255])), #white
                    
                ]
NOTHING = 0
BOARD_QUARTER = np.array([
                            [NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING],
                            [NOTHING, 2, NOTHING, NOTHING, NOTHING, 1, -1, NOTHING],
                            [NOTHING, NOTHING ,NOTHING, NOTHING, 1, -1, 1, NOTHING],
                            [NOTHING, NOTHING, NOTHING, 1, -1, 1, NOTHING ,NOTHING],
                            [NOTHING, NOTHING, 1, -1, 1, NOTHING, NOTHING, NOTHING],
                            [NOTHING, 1, -1, 1, NOTHING, NOTHING, NOTHING, NOTHING],
                            [NOTHING, -1, 1, NOTHING, NOTHING, NOTHING, 2, NOTHING],
                            [NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING]
                        ])

def orietation2board(orientation):
    boards = []
    
    for o in orientation:
        boards.append(BOARD_QUARTER if o == 0 else np.rot90(BOARD_QUARTER))
    
    row1 = np.concatenate([boards[0],boards[1]], axis=1)
    row2 = np.concatenate([boards[2],boards[3]], axis=1)
    
    return np.concatenate([row1, row2])

def scan_initial_board(board_img, bonus_board, scorer, shape_templates, id):
    for i in range(0, 16):
        for j in range(0, 16):
            if bonus_board[i][j] == -1:
                square = get_square(board_img, i+1, j+1)
                shape_code, color_code, score = identify_piece(square, shape_templates, id)
                scorer.board_state[i][j] = shape_code+color_code

def import_images(path, color=True):
    files = sorted(glob.glob(path))
    return [cv.imread(files[i], cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE) for i in tqdm(range(len(files)), desc="Importing images")]

def get_board(img):
    copy_img = deepcopy(img)
    copy_img = cv.cvtColor(copy_img, cv.COLOR_BGR2HSV)
    low = np.array([25, 10, 0])
    high = np.array([180, 255, 255])

    mask = cv.inRange(copy_img, low, high)

    mask_median_blur = cv.medianBlur(mask, 3)
    mask_gausian_blur = cv.GaussianBlur(mask_median_blur, (0,0), 5)
    mask_sharpened = cv.addWeighted(mask_median_blur, 1.2, mask_gausian_blur, -0.8, 0)
    _, thresh = cv.threshold(mask_sharpened, 70, 255, cv.THRESH_BINARY)
    
    # thresh = cv.bitwise_not(thresh)

    # kernel_erode = np.ones((25, 25), np.uint8)
    
    # thresh = cv.erode(thresh, kernel_erode)
    #show_img(thresh)
    #kernel_dilate = np.ones((13, 13), np.uint8)
    #thresh = cv.dilate(thresh, kernel_dilate)

    contours, _ = cv.findContours(thresh,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 1664
    height = 1664
    
    """
    top_right[0]+=32
    top_right[1]-=32
    bottom_left[0]-=32
    bottom_left[1]+=32
    """
    # print(top_left)
    #puzzle = np.array([[top_left-32,top_right,bottom_right+32,bottom_left]],dtype=np.float32)
    puzzle = np.array([[top_left,top_right,bottom_right,bottom_left]],dtype=np.float32)
    dest = np.array([[32,32],[width - 32, 32],[width - 32,height - 32],[32,height -32]],dtype=np.float32)

    M = cv.getPerspectiveTransform(puzzle,dest)
    result = cv.warpPerspective(img,M,(width,height))

    return result

# [1,1] colt sus
def get_square(img, l, c, p=0):
    return img[(l-1)*100 -p + 32:l*100+p+32, (c-1)*100-p+32:c*100+p+32].copy()

def clamp(n, minn, maxn):
    return max(minn, min(n, maxn))

def find_square(w, h):
    #print(w, h)
    return clamp((w-32)//100+1, 1, 16), clamp((h-32)//100+1, 1, 16)

def vote_for_square(img):
    mat = np.zeros((17, 17))
    indices = np.transpose(np.nonzero(img>=190))
    for h,w in indices:
        c, l =find_square(w, h)
        mat[l][c] += 1
    
    squares_over_tresh = np.transpose(np.nonzero(mat>=3000))
    return squares_over_tresh

def template_match(templates, square):
    max_score = 0
    maxt = 0
    ratios = [0.8, 0.85, 0.9, 0.95]
    for t in range(len(templates)):
        for ratio in ratios:
            temp = cv.resize(templates[t], (0,0), fx = ratio, fy = ratio)
            res = cv.matchTemplate(square, temp, cv.TM_CCORR_NORMED) 
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            if max_val > max_score:
                max_score = max_val
                maxt = t
    
    return maxt, max_score

def show_img(img):
    cv.imshow("Image", cv.resize(img, (640, 640)))
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_color(img, id):
    results = np.zeros(len(COLOR_FILTERS))
    
    cimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    M = 0
    # show_img(img)
    for i, filter in enumerate(COLOR_FILTERS):
        low, high = filter
        mask = cv.inRange(cimg, low, high)
        
        kernel1 = np.ones((3,3), np.uint8)
        eroded = cv.erode(mask, kernel1)
        dialated = cv.dilate(eroded, kernel1)
        
        results[i] = dialated.sum()
        if M < results[i]:
            M = results[i]
            img2s = dialated
    cv.imwrite(SAVE_TEMPLATES + f"template_{id}.jpg", img2s)
    return COLORS[results.argmax()], img2s

def get_frame_diff(initial, next):
    initial_gray = cv.cvtColor(initial, cv.COLOR_BGR2GRAY)
    next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
    
    diff = cv.absdiff(initial_gray, next_gray)

    diff_median_blur = cv.medianBlur(diff, 3)
    diff_gausian_blur = cv.GaussianBlur(diff_median_blur, (0,0), 5)
    diff_sharpened = cv.addWeighted(diff_median_blur, 1.2, diff_gausian_blur, -0.8, 0)

    _, diff_thresh = cv.threshold(diff_sharpened, 30, 255, cv.THRESH_BINARY)

    kernel1 = np.ones((7,7), np.uint8)
    kernel2 = np.ones((11,11), np.uint8)
    
    diff_eroded = cv.erode(diff_thresh, kernel1)
    diff_dialated = cv.dilate(diff_eroded, kernel2)
    #show_img(diff_thresh)
    #show_img(diff_eroded)
    return diff_dialated

def get_orientation(board):
    # 1 diag principala, 0 diag secundara 
    grey_board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
    orientation_list = []
    templates = import_images(PATH_TEMPLATES_2, color = False)
    squares_to_check = [
        [(2, 2), (2, 7), (7, 2), (7, 7)],
        [(2, 10), (2, 15), (7, 10), (7, 15)],
        [(10, 2), (10, 7), (15, 2), (15, 7)],
        [(10, 10), (10, 15), (15, 10), (15, 15)]
    ]
    
    for mini_board in squares_to_check:
        square0 = get_square(grey_board, mini_board[0][0], mini_board[0][1]) 
        square3 = get_square(grey_board, mini_board[3][0], mini_board[3][1]) 
        s1 = template_match(templates, square0)[1] + template_match(templates, square3)[1]
        square1 = get_square(grey_board, mini_board[1][0], mini_board[1][1]) 
        square2 = get_square(grey_board, mini_board[2][0], mini_board[2][1]) 
        s2 = template_match(templates, square1)[1] + template_match(templates, square2)[1]
        
        if s1 > s2: 
            orientation_list.append(0)
        else:
            orientation_list.append(1)
        #print(s1)
        #print(s2)
    return orientation_list 

# 1: cerc, 2: trifoi, 3: romb, 4: patrat, 5: stea 4 colturi, 6: stea 8 colturi
def load_shape_templates(path_to_templates):
    templates = []
    for i in range(1, 7):
        tmpl = cv.imread(f"{path_to_templates}/{i}.jpg", cv.IMREAD_GRAYSCALE)
        if tmpl is None:
            print(f"Warning: Template {i}.jpg not found!")
        templates.append(tmpl)
    return templates

def identify_piece(img_sq, templates, id):
    color_name, mask = get_color(img_sq, id) 
    color_code = COLOR_CODE_MAP.get(color_name, "?")
    
    # return indexul (0-5) adun 1 pentru a avea formele 1-6
    shape_idx, score = template_match(templates, mask)
    shape_code = str(shape_idx + 1)
    # print(shape_idx, score)
    return shape_code, color_code, score

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value