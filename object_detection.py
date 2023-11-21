import cv2
import numpy as np

net = cv2.dnn.readNet('yolov4-obj_last_new.weights', 'yolov4-obj_new.cfg')

classes = []
with open("custom.txt", "r") as f:
    classes = f.read().splitlines()


# for video capture
cap = cv2.VideoCapture(0)
# to load the image
# img = cv2.imread('7.jpg')
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # preparing the image
    # dividing by 255 pixels for normalising; 416x416 should be dimension
    # no mean subtraction so 0,0,0; converting to RGB so True and no croping
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)  # to set the input from blob in to the network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)   # multiplied by height/width as we had normalised earlier
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # to get positions of corners
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5, 0.4)  # 0.5 threshold value; 0.4: Non-Max_Suppression

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))   # 3 channels

    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)






    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if (key == 27):       # press on escape key to break loop
        break

cap.release()
cv2.destroyAllWindows()
