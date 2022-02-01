import cv2
import numpy as np
from time import time
import os
from PIL import Image
# import matplotlib.pyplot as plt


orb = cv2.ORB_create(nfeatures=500)
index_params = dict(algorithm =6, table_number = 6, key_size=12, multi_probe_level=12)
search_params ={}
flann = cv2.FlannBasedMatcher(index_params, search_params)



def filter_function(previous_frame, frame, prev_des, des, thresh=20, scale=0.75):
    
    temp1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp2 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    h,w = temp1.shape
    # temp1 = cv2.resize(temp1,None,fx=scale,fy=scale )
    # temp2 = cv2.resize(temp2,None,fx=scale,fy=scale )
    temp1 = temp1/(np.linalg.norm(temp1))
    temp2 = temp2/(np.linalg.norm(temp2))
    alpha = np.sum(np.abs(temp1-temp2))
    if alpha > thresh:
        # img = cv2.imread('images/scene.jpg')
        # gray_img = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # kp1, des1 = orb.detectAndCompute(previous_frame, None)
        # kp2, des2 = orb.detectAndCompute(frame, None)


        matches = flann.knnMatch(prev_des,des,k=2)

        good_matches = []
        for m,n in matches:
            if m.distance<0.75*n.distance:
                good_matches.append(m)
        
        print(len(good_matches), len(matches)) #need a more rigourous analysis. aise toh kuch malum nahi chalega.
        if(len(good_matches)/len(matches) > 0.6):

            # fig = plt.figure(figsize=(10, 7))
            # rows=2
            # columns=1
            # fig.add_subplot(rows,columns,1)
            # plt.axis('off')
            
            # plt.imshow(temp1, cmap='gray')
            # fig.add_subplot(rows,columns,2)
            # plt.axis('off')

            # plt.imshow(temp2, cmap='gray')
            
            # plt.show()
            return False
        
        return True
    else:
        return False


def vid_imgs(file_path, folder_name='D:\\temp_project\\', thresh=20, scale = 0.75):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("ERROR! Corrupted Video")
    ret, frame = cap.read()
    # frame_shape = frame.shape
    # prev_frame = frame
    prev_frame = cv2.resize(frame, None, fx=scale, fy = scale)
    _, des1 = orb.detectAndCompute(prev_frame, None)

    # prev_difference = 0
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # temp_list=[]
    start = time()
    slide_count = 0
    img_list=[]
    iter = 0
    while cap.isOpened():
        count += 1
        # print(count)
        # if count%14==0:
        #     print(count/14)
        ret, frame = cap.read()


        #this is for per second matching. there are around 3600 seconds in the recording and it is taking some more time. per 5 second processing? or per 3 second processing. Maybe some another heuristic I need to find.
        #this isnt the right thing. the slides might have minor changes u see. I think the procedure demands to analuse the type of change happening. Like particular to eliminate the damn zoom only, because the slides should 
        if ret and count%fps == 0:
            print(iter) 
            iter+=1
            frame_1 = cv2.resize(frame, None, fx = scale, fy=scale)
            _, des2 = orb.detectAndCompute(frame_1, None)
            
            should_we_select = filter_function(prev_frame, frame_1, des1, des2)

            
            if should_we_select:
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_list.append(frame)

                
                slide_count+=1
            des1 = des2
            prev_frame= frame_1
        elif not ret:
            break

        # print(count, "hello I am still here")
    end = time()
    cap.release()


    # print(len(temp_list))
    # print(make_histogram(temp_list))
    print("TOTAL TIME TAKEN", end-start)

    return img_list


def images2pdf(img_list,pdf_path,pdf_name):
    # if(img_list==None or len(img_list)==0):
    #     print("wrong data type, please check input carefully")
    #     return
    # print(img_list)
    if len(img_list)==1:
        img_list[0] = img_list[0].convert('RGB')
        img_list[0].save(pdf_path + pdf_name)
    img_list1 = [Image.fromarray(i) for i in img_list]
    # img_list1 = [i.convert('RGB') for i in img_list1]
    im1 = img_list1[0]
    im1.save(pdf_path+pdf_name,save_all=True, append_images = img_list1[1:])
    print("ALL IMAGES ARE SAVED SUCCESSFULLY AS PDF AT", pdf_path)


if __name__ == '__main__':
    # print(os.getcwd())
    # vid_trimmer("D:\\downloads\\Link for class wednesday 12-01-22-20220112_110314-Meeting Recording.mp4")
    temp = vid_imgs("D:\\downloads\\Link for class wednesday 12-01-22-20220112_110314-Meeting Recording.mp4")
    images2pdf(temp, "D:\\", "first1.pdf")
    
    