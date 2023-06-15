import cv2
import numpy as np
import time

start = time.time()
cap = cv2.VideoCapture('C:\\Users\\rahul\\PycharmProjects\\traffic\\cars13_AdobeExpress_AdobeExpress.mp4')
net = cv2.dnn.readNetFromONNX("C:\\Users\\rahul\\PycharmProjects\\traffic\\yolov5n.onnx")
print(net)
file = open("C:\\Users\\rahul\\PycharmProjects\\traffic\\coco.txt","r")
classes = file.read().split('\n')
print(classes)

# def detect_red_and_yellow(img, Threshold=0.01):
#     """
#     detect red and yellow
#     :param img:
#     :param Threshold:
#     :return:
#     """

#     desired_dim = (30, 90)  # width, height
#     img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

#     # lower mask (0-10)
#     lower_red = np.array([0, 70, 50])
#     upper_red = np.array([10, 255, 255])
#     mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

#     # upper mask (170-180)
#     lower_red1 = np.array([170, 70, 50])
#     upper_red1 = np.array([180, 255, 255])
#     mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

#     # defining the Range of yellow color
#     lower_yellow = np.array([21, 39, 64])
#     upper_yellow = np.array([40, 255, 255])
#     mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

#     # red pixels' mask
#     mask = mask0 + mask1 + mask2

#     # Compare the percentage of red values
#     rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

#     if rate > Threshold:
#         return True
#     else:
#         return False



while True:
    img = cap.read()[1]
    if img is None:
        break
    img = cv2.resize(img, (1000,600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11,11), 0)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold = 100, minLineLength=5, maxLineGap=250)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (y1 > 400 or y2 > 400): #Filter out the lines in the top of the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
    
    # print(detections)


    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
    # detect_red_and_yellow(img, Threshold=0.01)
    count = 0
    for i in indices:
        count = count + 1
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label 
        #+ "{:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
        cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        sum = 0
        # count = str(count)
        if(label=="car" or label=="bus" or label=="truck" or label=="bicycle"):
            print("Total number of vehicles: {}".format(count))
        cv2.putText(img,
				(50, 50),
				font, 1,
				(0, 255, 255),
				2,
				cv2.LINE_4)
        count = int(count)        

    cv2.imshow("",img)
    end = time.time()
    k = cv2.waitKey(1)
    if k == ord('q'):
        print("Time taken: {0:.2f} secs".format(end - start))
        break
    print("Total time taken: {0:.2f} secs".format(end - start))