import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # TODO ...
    A = np.zeros((9,))
    b = np.zeros((9,))
    for i in range(len(p1)):
        in_p = p2[i]; out_p = p1[i] 
        row1 = np.array([in_p[0],in_p[1],1,0,0,0,-out_p[0]*in_p[0],-out_p[0]*in_p[1],-out_p[0]])
        row2 = np.array([0,0,0,in_p[0],in_p[1],1,-out_p[1]*in_p[0],-out_p[1]*in_p[1],-out_p[1]])
        A = np.vstack([A,row1,row2])
    A = A[1:]
    
    _,_,v = np.linalg.svd(A)
    h = v[-1]
    H = h.reshape(3,3)
    
    return H

def compute_h_norm(p1, p2):
    # TODO ...
    p1=np.asarray(p1,dtype=np.float32); p2 = np.asarray(p2,dtype=np.float32)
    mx1=np.mean(p1[:,0]); my1=np.mean(p1[:,1])
    mx2=np.mean(p2[:,0]); my2=np.mean(p2[:,1])
    T = np.array([[1/mx2,0,-1],[0,1/my2,-1],[0,0,1]])
    T_ = np.array([[1/mx1,0,-1],[0,1/my1,-1],[0,0,1]])

    p1[:,0]/=mx1; p1[:,1]/=my1
    p2[:,0]/=mx2; p2[:,1]/=my2
    p1 -= 1; p2 -= 1
    H_norm = compute_h(p1, p2)

    H = np.linalg.inv(T_)@H_norm@T

    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    leftTop = H@np.array([0,0,1]); rightTop = H@np.array([400,0,1])
    leftBot = H@np.array([0,302,1]); rightBot = H@np.array([400,302,1])
    
    xs = np.array([leftTop[0]/leftTop[2],leftBot[0]/leftBot[2], rightTop[0]/rightTop[2],rightBot[0]/rightBot[2]])
    ys = np.array([leftTop[1]/leftTop[2], rightTop[1]/rightTop[2],leftBot[1]/leftBot[2], rightBot[1]/rightBot[2]])

    left = round(xs.min()); right = round(xs.max())
    top = round(ys.min()); bot = round(ys.max())

    x0 = left; y0 = top

    igs_warp = np.zeros((bot-top,right-left, 3))
    H_inv = np.linalg.inv(H)

    for y in range(igs_warp.shape[0]):
        for x in range(igs_warp.shape[1]):
            inpoint = H_inv@np.array([x+x0,y+y0,1])
            inx = inpoint[0]/inpoint[2]
            iny = inpoint[1]/inpoint[2]
            if inx >= 0 and inx <= igs_in.shape[1]-1 and iny >= 0 and iny <= igs_in.shape[0]-1:
                low_x = math.floor(inx); low_y = math.floor(iny)
                a = inx-low_x; b = iny-low_y
                igs_warp[y,x,:] += (1-a)*(1-b)*igs_in[low_y,low_x,:]
                igs_warp[y,x,:] += (1-a)*b*igs_in[low_y+1,low_x,:]
                igs_warp[y,x,:] += a*(1-b)*igs_in[low_y,low_x+1,:]
                igs_warp[y,x,:] += a*b*igs_in[low_y+1,low_x+1,:]
            
    x_changed = False; y_changed = False
    if top > 0:
        y_changed = True
    if left > 0:
        x_changed = True

    ym = min(0,top); xm = min(0,left)

    yrange = bot-ym; xrange=right-xm
    igs_merge = np.zeros((yrange,xrange,3))

    if x_changed and y_changed:
        igs_merge[y0:,x0:,:] = igs_warp.copy()
    elif x_changed:
        igs_merge[:,x0:,:] = igs_warp.copy()
    elif y_changed:
        igs_merge[y0:,:,:] = igs_warp.copy()
    else:
        igs_merge = igs_warp.copy()

    for ch in range(3):
        for y in range(igs_ref.shape[0]):
            for x in range(igs_ref.shape[1]):
                if igs_merge[y-ym,x-xm,ch] == 0:
                    igs_merge[y-ym,x-xm,ch] = igs_ref[y,x,ch]
    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...
    igs = np.asarray(igs, dtype=np.float64)
    H = compute_h_norm(p2, p1)
    H_inv = np.linalg.inv(H)
    igs_rec = np.zeros_like(igs)

    for y in range(igs_rec.shape[0]):
        for x in range(igs_rec.shape[1]):
            inpoint = H_inv@np.array([x,y,1])
            inx = inpoint[0]/inpoint[2]
            iny = inpoint[1]/inpoint[2]
            if inx >= 0 and inx <= igs.shape[1]-1 and iny >= 0 and iny <= igs.shape[0]-1:
                low_x = math.floor(inx); low_y = math.floor(iny)
                a = inx-low_x; b = iny-low_y
                igs_rec[y,x,:] += (1-a)*(1-b)*igs[low_y,low_x,:]
                igs_rec[y,x,:] += (1-a)*b*igs[low_y+1,low_x,:]
                igs_rec[y,x,:] += a*(1-b)*igs[low_y,low_x+1,:]
                igs_rec[y,x,:] += a*b*igs[low_y+1,low_x+1,:]

    return igs_rec

def set_cor_mosaic():
    # TODO ...
    p_in = np.array([[121,223],[265,138],[203,223],[383,137],[288,78],[251,197],[375,98],[163,197],[89,163],[276,70]])
    p_ref = np.array([[307,22],[281,137],[252,50],[162,206],[343,231],[236,83],[207,267],[303,48],[394,40],[371,235]])
    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    c_in = np.array([[156,11],[261,21],[156,264],[261,250]])
    c_ref = np.array([[150,35],[250,35],[150,241],[250,241]])
    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/wdc1.png').convert('RGB')
    img_ref = Image.open('data/wdc2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('wdc1_warped.png')
    img_merge.save('wdc_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
