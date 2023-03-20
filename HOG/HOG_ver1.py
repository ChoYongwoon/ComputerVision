from math import e
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # TODO: implement this function
    filter_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    filter_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    return filter_x, filter_y


def filter_image(im, filter):
    # TODO: implement this function
    # padding
    h, w = im.shape
    k = len(filter)
    k //= 2
    original_img = np.zeros((h+2*k,w+2*k))
    original_img[k:-k,k:-k] = im
    im_filtered = np.zeros((h,w))
    
    for y in range(h):
        for x in range(w):
            tmp = 0
            for i in range(2*k+1):
                for j in range(2*k+1):
                    tmp += filter[i,j]*original_img[y+2*k-i,x+2*k-j]
            im_filtered[y,x] = tmp

    return im_filtered

def get_gradient(im_dx, im_dy):
    # TODO: implement this function
    h, w = im_dx.shape
    grad_mag = np.sqrt(np.power(im_dx,2)+np.power(im_dy,2))
    grad_angle = np.arctan2(im_dy,im_dx)
    for y in range(h):
        for x in range(w):
            if grad_angle[y,x] < 0:
                grad_angle[y,x] += np.pi
    return grad_mag, grad_angle

def build_histogram(grad_mag, grad_angle, cell_size):
    # TODO: implement this function
    m, n = grad_mag.shape
    M = m//cell_size; N = n//cell_size
    ori_histo = np.zeros((M,N,6))

    for y in range(M*cell_size):
        for x in range(N*cell_size):
            in_y = y//cell_size; in_x = x//cell_size
            if grad_angle[y,x] > 165/180*np.pi or grad_angle[y,x] < 15/180*np.pi:
                ori_histo[in_y,in_x,0] += grad_mag[y,x]
            elif grad_angle[y,x] < 45/180*np.pi:
                ori_histo[in_y,in_x,1] += grad_mag[y,x]
            elif grad_angle[y,x] < 75/180*np.pi:
                ori_histo[in_y,in_x,2] += grad_mag[y,x]
            elif grad_angle[y,x] < 105/180*np.pi:
                ori_histo[in_y,in_x,3] += grad_mag[y,x]
            elif grad_angle[y,x] < 135/180*np.pi:
                ori_histo[in_y,in_x,4] += grad_mag[y,x]
            else:
                ori_histo[in_y,in_x,5] += grad_mag[y,x]

    return ori_histo

def get_block_descriptor(ori_histo, block_size):
    # TODO: implement this function
    M, N, C = ori_histo.shape
    B = block_size
    e = 0.001
    ori_histo_normalized = np.zeros((M-B+1,N-B+1,C*np.power(B,2)))

    for y in range(M-B+1):
        for x in range(N-B+1):
            norm = e**2
            for i in range(B):
                for j in range(B):
                    norm += np.dot(ori_histo[y+i,x+j,:],ori_histo[y+i,x+j,:])
                    ori_histo_normalized[y,x,i*B*C+j*C:i*B*C+j*C+C]=ori_histo[y+i,x+j,:]
            ori_histo_normalized[y,x,:] /= np.sqrt(norm)

    return ori_histo_normalized

# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    #plt.show()
    plt.savefig('hog.png')

def extract_hog(im,visualize=False,cell_size=8,block_size=2):
    # TODO: implement this function
    sobelx, sobely = get_differential_filter()
    im_dx = filter_image(im, sobelx); im_dy = filter_image(im, sobely)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    ori_histo = build_histogram(grad_mag,grad_angle,cell_size)
    hog = get_block_descriptor(ori_histo,block_size)

    if visualize:
        visualize_hog(im,hog,cell_size,block_size)
    return hog

def face_recognition(I_target, I_template):
    # TODO: implement this function
    h, w = I_target.shape
    h_t,w_t = I_template.shape

    hog_template = extract_hog(I_template)
    vector_template = hog_template.flatten()
    vec_tem = vector_template - np.mean(vector_template)
    vec_tem_norm = np.linalg.norm(vec_tem)
    
    sobelx, sobely = get_differential_filter()
    im_dx = filter_image(I_target, sobelx); im_dy = filter_image(I_target, sobely)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    scores = np.zeros((h-h_t+1,w-w_t+1))

    for y in range(h-h_t+1):
        for x in range(w-w_t+1):
            ori_histo = build_histogram(grad_mag[y:y+h_t+1,x:x+w_t+1], grad_angle[y:y+h_t+1,x:x+w_t+1],8)
            hog_cur = get_block_descriptor(ori_histo,2)
            cur_vec = hog_cur.flatten()
            vec_tar = cur_vec-np.mean(cur_vec)
            vec_tar_norm = np.linalg.norm(vec_tar)
            scores[y,x] = vec_tar.dot(vec_tem)/(vec_tem_norm*vec_tar_norm)
    
    bounding_boxes = []  
    
    while True:
        tmp_y, tmp_x = np.where(scores == scores.max())
        tmp_y = tmp_y[0]; tmp_x = tmp_x[0]
        if(scores[tmp_y,tmp_x] < 0.35):
            break
        bounding_boxes.append([tmp_x,tmp_y,scores[tmp_y,tmp_x]])
        scores[tmp_y,tmp_x] = 0  
        for y in range(h-h_t+1):
            for x in range(w-w_t+1):
                if abs(tmp_y - y) >= h_t or abs(tmp_x-x) >= h_t:
                    continue
                x_inter1 = max(tmp_x,x); y_inter1 = max(tmp_y,y)
                x_inter2 = min(tmp_x+h_t,x+w_t); y_inter2 = min(tmp_y+h_t,y+w_t)
                intersect = abs((x_inter2-x_inter1)*(y_inter2-y_inter1))
                union = 2*h_t**2 - intersect
                IOU = intersect/union
                if IOU >= 0.5:
                    scores[y,x] = 0

    bounding_boxes = np.array(bounding_boxes)
    return bounding_boxes

def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.imsave('result_face_detection.png', fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im, visualize=True)
    
    I_target= cv2.imread('target.png', 0) # MxN image

    I_template = cv2.imread('template.png', 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png') # MxN image (just for visualization)
    
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0]) # visualization code
    
