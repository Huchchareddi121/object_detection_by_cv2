#To run this script use give argument like below line 
#python object_detection.py --video video_path --output output_video_path 

#Import the neccesary libraries


import numpy as np
import argparse
import cv2 
import time
import os

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or ')
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--output", default="output.avi",
                                  help='Path to output video file : '
                                       'To save output file  ' )
args = parser.parse_args()


video_path = args.video
#print(video_path)

input_filename,input_extention=os.path.splitext(video_path)
input_path ,input_file = os.path.split(video_path)
input_file,Not_required =os.path.splitext(input_file)
video_file_path=None
if input_path=='/' or input_path=="\\" :
    video_file_path= input_file+input_extention
elif input_path:
    video_file_path= video_path
else:
    video_file_path=video_path
    



text_filename='output.txt'
if os.path.exists(text_filename):
    append_write = 'a' # append if already exists
else:
    append_write = 'w'

text_file = open(text_filename,append_write)

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open video file or capture device. 
if args.video:
    #video_file_path
    #cap = cv2.VideoCapture(args.video)
    cap = cv2.VideoCapture(video_file_path)
else:
    cap = cv2.VideoCapture(0)

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
new_size = (frame_width, frame_height)
original_fps = cap.get(cv2.CAP_PROP_FPS)
output_video_path = args.output
#print(output_video_path)
filename,extention=os.path.splitext(output_video_path)
path ,file = os.path.split(output_video_path)
file,Not_required =os.path.splitext(file)
#isFile = os.path.isfile(output_video_path)
#print(isFile)
#print('path',path)
new_video=None
if extention=='.avi':
    #print(output_video_path)
    new_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), original_fps, new_size)
else:
    if path=='/' or path =="\\" :
        
        output_video_path= file+'.avi'#
        #print(output_video_path)
        new_video= cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), original_fps, new_size)
    elif path :
        if not os.path.exists(path):
            os.makedirs(path)
        output_video_path= path+'/'+file+'.avi'#
        #print(output_video_path)
        new_video= cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), original_fps, new_size)
    else:
        output_video_path= path+file+'.avi'
        new_video= cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), original_fps, new_size)
start = time.time()
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret :  
        
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols ,rows = frame_resized.shape[1] ,frame_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom , yLeftBottom , xRightTop , yRightTop = int(detections[0, 0, i, 3] * cols) ,int(detections[0, 0, i, 4] * rows),int(detections[0, 0, i, 5] * cols),int(detections[0, 0, i, 6] * rows)

                # Factor for scale to original size of frame
                heightFactor , widthFactor= frame.shape[0]/300.0 ,frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom , yLeftBottom, xRightTop, yRightTop= int(widthFactor * xLeftBottom) ,int(heightFactor * yLeftBottom), int(widthFactor * xRightTop),int(heightFactor * yRightTop) 
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))

                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),(xLeftBottom + labelSize[0], yLeftBottom + baseLine),(255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        new_video.write(frame) 
        cv2.imshow("frame", frame)
    else :
        break
    
    
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
new_video.release()
end = time.time()
seconds = end - start
print ("Time taken to process the video: {0} seconds".format(seconds))
total_time_taken="Time taken to process the video : {0} seconds".format(seconds)
fps  = 1 / seconds
print("Estimated frames per second : {0}".format(fps))
fps_calculated="Estimated frames per second : {0}".format(fps)
text_file.write(total_time_taken+'\n'+fps_calculated+'\n')
text_file.close()


