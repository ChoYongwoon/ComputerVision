import math
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# parameters

datadir = './inputs'
resultdir='./outputs'

# you can calibrate these parameters
sigma=0.5
threshold=0.1
rhoRes=1
thetaRes=math.pi/360
nLines=20

def padding(Igs, k):
    h = len(Igs); w = len(Igs[0])
    out_img = np.zeros((h+2*k, w+2*k))
    out_img[k:-k,k:-k] = Igs.copy()
    for i in range(k):
        out_img[i] = out_img[k].copy()
        out_img[i+h+k] = out_img[k+h-1,:].copy()
        out_img[:,i] = out_img[:,k].copy()
        out_img[:,i+w+k] = out_img[:,k+w-1].copy()
    
    return out_img

def ConvFilter(Igs, G):
    # Padding
    h = len(Igs); w = len(Igs[0])
    k = len(G)
    k = k//2
    Iconv = padding(Igs, k)
    out_img = padding(Igs, k)
    # Convolution
    for i in range(h):
        for j in range(w):
            tmp_val = 0
            for m in range(2*k+1):
                for n in range(2*k+1):
                    tmp_val += G[m,n]*Iconv[i+2*k-m, j+2*k-n]
            out_img[i+k,j+k] = tmp_val
    out_img = out_img[k:-k, k:-k].copy()
    return out_img

def gaussianSmooth(Igs, k, sigma):
    kernel = np.zeros((2*k+1,2*k+1))
    for i in range(2*k+1):
        for j in range(2*k+1):
            x = k-i; y = k-j;
            kernel[i,j] = math.exp(-(x**2 + y**2)/(2*sigma**2))/(2*math.pi*sigma**2)
    total = sum(sum(kernel))
    kernel /= total
    smooth_img = ConvFilter(Igs, kernel)
    return smooth_img

def nms(Im, Io):
    h, w = Im.shape
    Im = padding(Im,1)
    for i in range(1,h+1):
        for j in range(1,w+1):
            angle = Io[i-1,j-1]
            if angle < 0:
                angle += 180
                
            if angle < 22.5 or angle >= 157.5:
                if Im[i,j] < Im[i,j+1] or Im[i,j] < Im[i,j-1]:
                    Im[i,j] = 0
            elif angle < 67.5:
                #if Im[i,j] < Im[i+1,j-1] or Im[i,j] < Im[i-1,j+1]:
                if Im[i,j] < Im[i+1,j] or Im[i,j] < Im[i-1,j]:
                    Im[i,j] = 0
            elif angle < 112.5:
                #if Im[i,j] < Im[i+1,j] or Im[i,j] < Im[i-1,j]:
                if Im[i,j] < Im[i+1,j-1] or Im[i,j] < Im[i-1,j+1]:
                    Im[i,j] = 0
            elif angle < 157.5:
                if Im[i,j] < Im[i-1,j-1] or Im[i,j] < Im[i+1,j+1]:
                    Im[i,j] = 0   
            else:
                Im[i,j] = 0
                
    Im = Im[1:-1,1:-1].copy()
    return Im

def EdgeDetection(Igs, sigma):
    h = len(Igs); w = len(Igs[0]);
    
    sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    sobely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
    
    smoothe_img = gaussianSmooth(Igs.copy(), 1, sigma)
    Ix = ConvFilter(smoothe_img,sobelx)
    Iy = ConvFilter(smoothe_img,sobely)
    Io = np.zeros((h,w))
    Io = np.arctan2(Iy,Ix)*180/math.pi
    Im = np.zeros((h,w), np.float64)
    
    for i in range(h):
        for j in range(w):
            Im[i,j] = math.sqrt(math.pow(Ix[i,j],2)+math.pow(Iy[i,j],2))
            
    Im = nms(Im, Io)
    Im /= Im.max()
    return Im, Io, Ix, Iy

def HoughTransform(Im,threshold, rhoRes, thetaRes):
    h, w = Im.shape
    theta_max = 2*math.pi
    rho_max = math.ceil(math.sqrt(math.pow(h,2)+math.pow(w,2)))
    rho = np.arange(0,rho_max+rhoRes,rhoRes)
    theta = np.arange(0,theta_max+thetaRes,thetaRes)
    H = np.zeros((len(rho), len(theta)))
    
    for y in range(h):
        for x in range(w):
            if Im[y,x] > threshold:
                for k in range(len(theta)):
                    
                    r = x*np.cos(theta[k])+y*np.sin(theta[k])
                    r_low = int(r//rhoRes)
                    if r_low < rho_max and r_low >=0:
                        if r > 0:
                            H[r_low,k] += (r_low + 1)*rhoRes - r
                            H[r_low+1,k] += r - r_low*rhoRes
                        else:
                            H[r_low,k] += r-(r_low-1)*rhoRes
                            H[r_low-1,k] += r_low*rhoRes-r
                    
    H /= H.max()
    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    H_ = H.copy()
    h,w = H_.shape
    lRho = np.zeros((nLines,)); lTheta = np.zeros((nLines,))
    cnt = 0
    rho_thres = max(np.nonzero(H)[0])/40
    
    while cnt < nLines:
        y, x = np.where(H_ == H_.max())
        if type(y) != int:
            y, x = y[0], x[0]

        tmpRho = y*rhoRes
        tmpTheta = x*thetaRes
        theta_degree = tmpTheta*180/math.pi
        check = False
        for i in range(len(lTheta)):
            if abs(lTheta[i]*180/math.pi - theta_degree) < 15 or abs(lTheta[i]*180/math.pi - theta_degree) > 345:
                if abs(lRho[i] - tmpRho) <= rho_thres:
                    check = True
                    break
                    
        H_[y,x] = 0
        
        if check:
            continue
        else:
            lRho[cnt] = tmpRho
            lTheta[cnt] = tmpTheta
            cnt+= 1

    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im, threshold):
    dx = [-1,0,1,-1,0,1,-1,0,1]
    dy = [-1,-1,-1,0,0,0,1,1,1]
    l = {}
    tmp = []
    h, w = Im.shape
    for i in range(len(lTheta)):
        rho = lRho[i]
        theta = lTheta[i]
        line = []
        tmp_line = []
        if np.sin(theta) != 0:
            for x in range(2,w-2):
                y = round(rho/math.sin(theta) - x/math.tan(theta))
                if(y < h-2 and y >0):
                    check = False
                    for j in range(len(dx)):
                        if Im[y+dy[j], x+dx[j]] > threshold:
                            check = True
                    if check:
                        tmp_line.append([y,x])
                    else:
                        if len(tmp_line) != 0:
                            if len(line) == 0:
                                line = tmp_line
                                tmp_line = []
                            elif len(line) != 0:
                                if (len(tmp_line)) > len(line):
                                    line = tmp_line
                                tmp_line = []
                                    
        elif np.cos(theta) != 0:
            for y in range(2,h-2):
                x = round(rho/math.cos(theta) - y*math.tan(theta))
                if(x < w-2 and x >0):
                    check = False
                    for j in range(len(dx)):
                        if Im[y+dy[j],x+dx[j]] > threshold:
                            check = True
                    if check:
                        tmp_line.append([y,x])
                    else:
                        if len(tmp_line) != 0:
                            if len(line) == 0:
                                line = tmp_line
                                tmp_line = []
                            elif len(line) != 0:
                                if (len(tmp_line)) > len(line):# and tmp_avg > Im_avg:
                                    line = tmp_line
                                tmp_line = []
        if len(tmp_line) > len(line):
            line = tmp_line
            
        if len(line) == 0:
            if len(tmp_line) == 0:
                continue
            else:
                st = tmp_line[0]
                en = tmp_line[-1]
        else:
            st = line[0]
            en = line[-1]
        l[i] = {'start':st, 'end':en}

    return l

def main():
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        tmp_list = img_path.split('.')
        img_num = 0
        for st in tmp_list:
            if 'img' in st:
                idx = st.index('img')
                idx += 3
                img_num = st[idx:]
                
        original_img = Image.open(img_path).convert("RGB")
        img = Image.open(img_path).convert("L")
        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im, threshold)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
        plt.figure()
        plt.imshow(Im,cmap='gray')
        plt.axis('off')
        plt.savefig(resultdir +'/img'+ img_num +'_Im.jpeg')
        
        plt.figure()
        plt.imshow(H,cmap='gray')
        plt.axis('off')
        plt.savefig(resultdir +'/img'+ img_num +'_H.jpeg')

        tmp = []
        h, w = Im.shape
        for i in range(len(lTheta)):
            rho = lRho[i]
            theta = lTheta[i]
            line = []
            tmp_line = []
            if np.sin(theta) != 0:
                for x in range(w):
                    y = round(rho/math.sin(theta) - x/math.tan(theta))
                    if(y < h and y >=0):
                        tmp_line.append([y,x])
            elif np.cos(theta) != 0:
                for y in range(h):
                    x = round(rho/math.cos(theta) - y*math.tan(theta))
                    if(x < w and x >=0):
                        tmp_line.append([y,x])
            tmp.append(tmp_line)
        
        plt.figure()
        plt.imshow(original_img)
        plt.axis('off')
        for i in range(len(tmp)):
            start = tmp[i][0]
            end = tmp[i][-1]
            plt.plot([start[1],end[1]],[start[0],end[0]])
        plt.savefig(resultdir +'/img'+ img_num +'_Line.jpeg')

        plt.figure()
        plt.imshow(original_img)
        plt.axis('off')
        for segment in l:
            start = l[segment]['start']
            end = l[segment]['end']
            plt.plot([start[1],end[1]],[start[0],end[0]], linewidth = 2, marker = '.')
        plt.savefig(resultdir + '/img'+img_num+'_LineSegments.jpeg')
        

if __name__ == '__main__':
    main()
