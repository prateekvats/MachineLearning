import tensorflow as tf
import numpy as np
import os,glob,cv2
import time

import sys,argparse


# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path='test'#sys.argv[1]
filename = dir_path +'/' +image_path
test_images= glob.glob(image_path+"/*")
image_size=64
num_channels=3



def getWindows(gridSize,x_begin,y_begin,x_end,y_end):
    windows=[]
    x_start= x_begin
    y_start= y_begin
    for gs in gridSize:
        for r in range(y_start,y_end,gs):
            for c in range(x_start,x_end,gs):
                window = []
                window.append(c)
                window.append(r)
                window.append(gs)
                windows.append(window)
        y_start = y_begin
        x_start = x_begin
    return windows


def selectWindows(image,windows,graph,sess):
    filteredWindows=[]
    h,w,c=image.shape
    start = time.time()
    for window in windows:
        crop_img = image[window[1]:window[1] + window[2], window[0]:window[0] + window[2]]
        if(predict(crop_img,sess,graph)):
            filteredWindows.append(window)
    end = time.time()
    print("Time to predict:",end-start)
    return filteredWindows

def drawWindows(image,windows):
    resultImage=image
    for window in windows:
        cv2.rectangle(resultImage, (window[0], window[1]), (window[0] + window[2], window[1] + window[2]), (0, 0, 255), 2)
    return resultImage


def getTFSession():
    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('vehicle-detection-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    return sess,saver,graph

def predict(image,sess,graph):
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images = []
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)



    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]

    if result[0][0]>0.95:
        return True
    else:
        return False
#
# counter=0
# sess,saver,graph = getTFSession()
# for im in test_images:
#     # Reading the image using OpenCV
#
#
#     image = cv2.imread(im)
#     h,w,c = image.shape
#     windows=getWindows(image,10,[32,64,128],int(w/5),int(h/2))
#     print("Total Windows:",len(windows))
#     # imageWithAllWindows = drawWindows(image,windows)
#     # cv2.imwrite("./output/output_allWindows"+str(counter)+".png",imageWithAllWindows)
#
#     windows = selectWindows(image,windows,graph,sess)
#     print("Selected Windows:", len(windows))
#     imageWithSelectedWindows = drawWindows(image,windows)
#     cv2.imwrite("./output/output"+str(counter)+".png",imageWithSelectedWindows)
#     counter+=1

def videoReading(videoFileName):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(videoFileName)
    sess, saver, graph = getTFSession()
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame

            frame = cv2.resize(frame, (1024, 576), 0, 0, cv2.INTER_LINEAR)
            h,w,c = frame.shape
            windows = getWindows([64], 100, 250,800,575)
            # processedFrame = drawWindows(frame, windows)
            selectedWindows = selectWindows(frame,windows,graph,sess)
            processedFrame= drawWindows(frame,selectedWindows)
            cv2.imshow('Frame', processedFrame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


videoReading('test/dashcam_video.mp4')
