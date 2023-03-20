import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
from tqdm import tqdm

def lucas_kanade_affine(T, I):
    # These codes are for calculating image gradient by using sobel filter
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this
    
    p = np.zeros((6,)) # initializer p
    
    ### START CODE HERE ###
    # [Caution] You should use only numpy and RectBivariateSpline functions
    # Never use opencv
    height = I.shape[0]; width = I.shape[1] 
    h = np.arange(0,I.shape[0],1)
    w = np.arange(0,I.shape[1],1)
    I = RectBivariateSpline(h,w,I)
    Ix = RectBivariateSpline(h,w,Ix)
    Iy = RectBivariateSpline(h,w,Iy)

    for i in range(300): # number of iter to calculate delta p
        H = np.zeros((6,6)) # Hessian matrix
        sum_mid = np.zeros((6,)) # sum of gradient*difference
        for y in h[::10]: # 1/10 subsampling
            for x in w[::10]: # 1/10 subsampling 
                wx = (1+p[0])*x+p[2]*y+p[4] # wx from p and (x, y)
                wy = p[1]*x+(1+p[3])*y+p[5] # wy from p and (x, y)
                if (wx < 0 or wx >= width or wy < 0 or wy >= height): # out of range -> continue
                    continue
                gx = Ix.ev(wy,wx) # Ix(W(x;p))
                gy = Iy.ev(wy,wx) # Iy(W(x;p))
                diff = T[y,x] - I.ev(wy,wx) # T - I(W(x;p)) a.k.a. diff
                gIjW = np.array([gx*x,gy*x,gx*y,gy*y,gx,gy]) # gradientI(W(x;p))T@(aW/ap) a.k.a. gIjW
                sum_mid += gIjW.T*diff # summation diff*calculate gIjW.T
                H += np.einsum('i,j -> ij', gIjW.T,gIjW) # summation gIjW.T @ gIjW
        H_inv = np.linalg.pinv(H) # calculate H^-1
        dp = np.einsum('ik, k -> i', H_inv, sum_mid) # calculate delta p
        p += dp # update p
    ### END CODE HERE ###
    
    return p
    
def subtract_dominant_motion(It, It1):
    
    ### START CODE HERE ###
    # [Caution] You should use only numpy and RectBivariateSpline functions
    # Never use opencv
    height = It.shape[0]; width = It.shape[1]
    p = lucas_kanade_affine(It1, It) # calculate affine matrix P
    h = np.arange(0,height,1) 
    w = np.arange(0,width,1)
    It = RectBivariateSpline(h,w,It)
    motion_image = np.zeros_like(It1)
    for y in h:
        for x in w:
            # calculate W(x;p)
            wx = (1+p[0])*x+p[2]*y+p[4]
            wy = p[1]*x+(1+p[3])*y+p[5]
            if (wx < 0 or wx >= width or wy < 0 or wy >= height):
                continue
            value = float(It.ev(wy,wx)) # I(W(x;p))
            if (value >= 0 and value <= 255):
                motion_image[y][x] = abs(It1[y][x] - value) # calculate difference : T-I(W(x;p)) 
    ### START CODE HERE ###
    
    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.05 * 256 # you can modify this
    
    mask = apply_hysteresis_threshold(motion_image, th_lo, th_hi)
    
    return mask

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    data_dir = 'data/motion'
    video_path = 'results/motion_best.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
    img_path = os.path.join(data_dir, "{}.jpg".format(0))
    It = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    for i in tqdm(range(1, 61)):
        img_path = os.path.join(data_dir, "{}.jpg".format(i))
        It1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        It_clone = It.copy()
        mask = subtract_dominant_motion(It, It1)
        It_clone = cv2.cvtColor(It_clone, cv2.COLOR_GRAY2BGR)
        It_clone[mask, 2] = 255 # set color red for masked pixels
        out.write(It_clone)
        It = It1
    out.release()