from __future__ import print_function
from scipy.spatial import distance as dist
from cv2 import cv2
from collections import OrderedDict
import argparse
import numpy as np
import itertools
import math
import imutils
import dlib
from skimage import io
from datetime import datetime
from datetime import date
import pandas as pd
import smtplib
import ssl
import email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import base64
import os



class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False


class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects

            


writer = None

W = None
H = None

carData = pd.DataFrame()

carData['color'] = []
carData['size'] = []
carData['time'] = []

net = cv2.dnn.readNetFromCaffe('C:\\Users\\19732\\Ex_Files_Learning_Python\\Exercise Files\\people-counting-opencv\\mobilenet_ssd\\MobileNetSSD_deploy.prototxt', 
'C:\\Users\\19732\\Ex_Files_Learning_Python\\Exercise Files\\people-counting-opencv\\mobilenet_ssd\\MobileNetSSD_deploy.caffemodel')

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalCars = 0



backSub = cv2.createBackgroundSubtractorMOG2()

capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('C:\\Users\\19732\\College Fall 2020\\Senior Project\\aerial_cars_driving_overpass.mp4'))


# # Old BGR
# colorBorders = [
# ([17, 15, 120], [18, 26, 237]), # RED
# ([63, 2, 3], [237, 88, 42]), # BLUE
# ([25, 148, 193], [28, 239, 236]), # YELLOW
# ([99, 84, 64], [155, 142, 136]), # GREY
# ([0, 0, 0], [98, 83, 63]), # BLACK
# ([201, 201, 214], [255, 255, 255]), # WHITE
# ([15, 100, 17], [18, 203, 26]), # GREEN
# ([8, 82, 201], [57, 131, 249]) # ORANGE
# ]

# New RGB
colorBorders = [
([120, 17, 15], [237, 26, 18]), # RED
([3, 2, 63], [42, 88, 237]), # BLUE
([193, 148, 25], [236, 239, 28]), # YELLOW
([64, 84, 99], [136, 142, 155]), # GREY
([0, 0, 0], [63, 83, 98]), # BLACK
([214, 201, 201], [255, 255, 255]), # WHITE
([17, 100, 15], [26, 203, 18]), # GREEN
([201, 82, 8], [249, 131, 57]) # ORANGE
]



while True:
    #read each frame
    ret, frame = capture.read()
    if frame is None:
        break
 

    # Resize frame for easier processing
    #frame = imutils.resize(frame, width=500)
    # frame = cv2.resize(frame, (1248, 702))

    # Change color scheme from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set Width / Height dimensions if empty
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    # Record the output video and store it on my computer
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer  = cv2.VideoWriter('C:\\Users\\19732\\College Fall 2020\\Senior Project', fourcc, 30,
    #     (W,H), True)


    status = 'waiting'
    rects = []

    # If we land on a tracking frame, process the image, and determine
    # where the blobs are
    if totalFrames % 20 == 0:
        status = 'detecting'
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

		# loop over the detections
        for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
            confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
            if confidence > 0.4:
				# extract the index of the class label from the
				# detections list
                idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
                # if CLASSES[idx] != "person":
                #     continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
                trackers.append(tracker)


    else:

        # track objects instead of detecting new ones
        for t in trackers:
            status = 'tracking'

            # update the tracker and its position
            t.update(rgb)
            pos = t.get_position()

            # unpack boundaries of square around object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # append these values to the rects list
            rects.append((startX, startY, endX, endY))

    cv2.line(frame, (round(W // 3), 0), (round(W // 3), H), (0, 255, 255), 1)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
            # Horizontal ↓
            x = [c[0] for c in to.centroids]
            direction = centroid[1] - np.mean(x)
            to.centroids.append(centroid)

            # Vertical ↓
            # y = [c[1] for c in to.centroids]
            # direction = centroid[1] - np.mean(y)
            # to.centroids.append(centroid)

			# check to see if the object has been counted or not
            if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
                # if direction < 0: # and centroid[1] < H // 2:
                #     totalUp += 1
                #     to.counted = True
                # if centroid[0] > round(W // 6) and direction > 0:


                #     dataToAdd = []
                #     # increment total car count
                #     totalCars += 1

                #     # render the color of each car

                #     cropped = frame[centroid[1]-10:centroid[1]+10, centroid[0]-10:centroid[0]+10]


                #     pixels = np.float32(cropped.reshape(-1, 3))


                #     n_colors = 1
                #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                #     flags = cv2.KMEANS_RANDOM_CENTERS

                #     _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                #     _, counts = np.unique(labels, return_counts=True)

                #     dominant = palette[np.argmax(counts)]

                
                #     #print(dominant)
				# 	# RENDER THE COLOR FROM RGB TO WORDS

                #     if dominant[0] >=139 and dominant[0] <= 255 and dominant[1] >= 0 and dominant[1] <= 160 and dominant[2] >= 0 and dominant[2] <= 128 and len(dataToAdd) == 0:
                #         dataToAdd.append('red')
                #     elif dominant[0] >=0 and dominant[0] <= 173 and dominant[1] >= 130 and dominant[1] <= 216 and dominant[2] >= 160 and dominant[2] <= 255 and len(dataToAdd) == 0:
                #         dataToAdd.append('blue')
                #     elif dominant[0] >=128 and dominant[0] <= 255 and dominant[1] >= 134 and dominant[1] <= 255 and dominant[2] >= 0 and dominant[2] <= 170 and len(dataToAdd) == 0:
                #         dataToAdd.append('yellow')
                #     elif dominant[0] >=105 and dominant[0] <= 245 and dominant[1] >= 105 and dominant[1] <= 245 and dominant[2] >= 105 and dominant[2] <= 245 and len(dataToAdd) == 0:
                #         dataToAdd.append('grey')
                #     elif dominant[0] >=0 and dominant[0] <= 90 and dominant[1] >= 0 and dominant[1] <= 90 and dominant[2] >= 0 and dominant[2] <= 98 and len(dataToAdd) == 0:
                #         dataToAdd.append('black')
                #     elif dominant[0] >=214 and dominant[0] <= 255 and dominant[1] >= 201 and dominant[1] <= 255 and dominant[2] >= 201 and dominant[2] <= 255 and len(dataToAdd) == 0:
                #         dataToAdd.append('white')
                #     elif dominant[0] >=17 and dominant[0] <= 26 and dominant[1] >= 100 and dominant[1] <= 303 and dominant[2] >= 15 and dominant[2] <= 18 and len(dataToAdd) == 0:
                #         dataToAdd.append('green')
                #     elif dominant[0] >=201 and dominant[0] <= 249 and dominant[1] >= 82 and dominant[1] <= 131 and dominant[2] >= 8 and dominant[2] <= 57 and len(dataToAdd) == 0:
                #         dataToAdd.append('orange')
                #     elif len(dataToAdd) == 0:
                #         dataToAdd.append('color not recognized')

          
                #     # Rendering Sizes

                #     #background subtraction + conversion to grayscale
                #     fgMask = backSub.apply(frame)

                #     # Gaussian blur ??
                #     fgMaskGauss = cv2.GaussianBlur(fgMask, (5,5), 0)

                #     # Converting to Black and White
                #     (thresh, fgMaskBW) = cv2.threshold(fgMaskGauss, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                #     # Getting contours
                #     contours, hierarchy = cv2.findContours(fgMaskBW, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                #     filteredCon = [] 

                #     for con in contours:
                #         if cv2.contourArea(con) > 250:
                #             filteredCon.append(con)
                #     # Calculate the centroids for each contour
                #     for c in filteredCon:
                #         M = cv2.moments(c)

                    
                #     # From moments, calculate centroids

                #         cX = int(M["m10"] / M["m00"])
                #         cY = int(M["m01"] / M["m00"])

                #         if cX <= centroid[0]+30 and cX >= centroid[0]-30 and cY <= centroid[1]+30 and cY >= centroid[1]-30:
                            
                #             if M['m00'] <= 1500 and len(dataToAdd) == 1:
                #                 dataToAdd.append('small')
                #             elif M['m00'] <= 25000 and len(dataToAdd) == 1:
                #                 dataToAdd.append('medium')
                #             elif len(dataToAdd) == 1:
                #                 dataToAdd.append('large')
                        
                #     if len(dataToAdd) == 1:
                #         dataToAdd.append('size not recognized')

                #     # RENDER THE TIME
                #     now = datetime.now()

                #     current_time = now.strftime("%H:%M:%S")

                #     timeStr = str(current_time)

                #     dataToAdd.append(timeStr)


                #     df_length = len(carData)
                #     carData.loc[df_length] = dataToAdd


                #     to.counted = True



				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
                if centroid[0] < round(W // 3):


                    dataToAdd = []
                    # increment total car count
                    totalCars += 1

                    # render the color of each car

                    cropped = frame[centroid[1]-10:centroid[1]+10, centroid[0]-10:centroid[0]+10]


                    pixels = np.float32(cropped.reshape(-1, 3))


                    n_colors = 1
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                    flags = cv2.KMEANS_RANDOM_CENTERS

                    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                    _, counts = np.unique(labels, return_counts=True)

                    dominant = palette[np.argmax(counts)]

                
                    #print(dominant)
					# RENDER THE COLOR FROM RGB TO WORDS

                    if dominant[0] >=139 and dominant[0] <= 255 and dominant[1] >= 0 and dominant[1] <= 160 and dominant[2] >= 0 and dominant[2] <= 128 and len(dataToAdd) == 0:
                        dataToAdd.append('red')
                    elif dominant[0] >=0 and dominant[0] <= 173 and dominant[1] >= 130 and dominant[1] <= 216 and dominant[2] >= 160 and dominant[2] <= 255 and len(dataToAdd) == 0:
                        dataToAdd.append('blue')
                    elif dominant[0] >=128 and dominant[0] <= 255 and dominant[1] >= 134 and dominant[1] <= 255 and dominant[2] >= 0 and dominant[2] <= 170 and len(dataToAdd) == 0:
                        dataToAdd.append('yellow')
                    elif dominant[0] >=105 and dominant[0] <= 245 and dominant[1] >= 105 and dominant[1] <= 245 and dominant[2] >= 105 and dominant[2] <= 245 and len(dataToAdd) == 0:
                        dataToAdd.append('grey')
                    elif dominant[0] >=0 and dominant[0] <= 90 and dominant[1] >= 0 and dominant[1] <= 90 and dominant[2] >= 0 and dominant[2] <= 98 and len(dataToAdd) == 0:
                        dataToAdd.append('black')
                    elif dominant[0] >=214 and dominant[0] <= 255 and dominant[1] >= 201 and dominant[1] <= 255 and dominant[2] >= 201 and dominant[2] <= 255 and len(dataToAdd) == 0:
                        dataToAdd.append('white')
                    elif dominant[0] >=17 and dominant[0] <= 26 and dominant[1] >= 100 and dominant[1] <= 303 and dominant[2] >= 15 and dominant[2] <= 18 and len(dataToAdd) == 0:
                        dataToAdd.append('green')
                    elif dominant[0] >=201 and dominant[0] <= 249 and dominant[1] >= 82 and dominant[1] <= 131 and dominant[2] >= 8 and dominant[2] <= 57 and len(dataToAdd) == 0:
                        dataToAdd.append('orange')
                    elif len(dataToAdd) == 0:
                        dataToAdd.append('color not recognized')

          
                    # Rendering Sizes

                    #background subtraction + conversion to grayscale
                    fgMask = backSub.apply(frame)

                    # Gaussian blur ??
                    fgMaskGauss = cv2.GaussianBlur(fgMask, (5,5), 0)

                    # Converting to Black and White
                    (thresh, fgMaskBW) = cv2.threshold(fgMaskGauss, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    # Getting contours
                    contours, hierarchy = cv2.findContours(fgMaskBW, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    filteredCon = [] 

                    for con in contours:
                        if cv2.contourArea(con) > 250:
                            filteredCon.append(con)
                    # Calculate the centroids for each contour
                    for c in filteredCon:
                        M = cv2.moments(c)

                    
                    # From moments, calculate centroids

                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        if cX <= centroid[0]+30 and cX >= centroid[0]-30 and cY <= centroid[1]+30 and cY >= centroid[1]-30:
                            
                            if M['m00'] <= 1500 and len(dataToAdd) == 1:
                                dataToAdd.append('small')
                            elif M['m00'] <= 25000 and len(dataToAdd) == 1:
                                dataToAdd.append('medium')
                            elif len(dataToAdd) == 1:
                                dataToAdd.append('large')
                        
                    if len(dataToAdd) == 1:
                        dataToAdd.append('size not recognized')

                    # RENDER THE TIME
                    now = datetime.now()

                    current_time = now.strftime("%H:%M:%S")

                    timeStr = str(current_time)

                    dataToAdd.append(timeStr)


                    df_length = len(carData)
                    carData.loc[df_length] = dataToAdd


                    to.counted = True

		# store the trackable object in our dictionary
        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    
    # big rectangle
    # cv2.rectangle(frame, (10, 30), (975,125), (255,255,255), -1)
    # cv2.putText(frame, 'Total Number of Vehicles: ' + str(totalCars), (15, 100),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)

    # Smaller one
    cv2.rectangle(frame, (10, 10), (100,30), (255,255,255), -1)
    cv2.putText(frame, str(totalCars), (15, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
	
    now = datetime.now()

    if now.strftime("%H:%M:%S") == "00:00:00":

        today = date.today()
        carData.to_csv('C:\\Users\\19732\\Ex_Files_Learning_Python\\Exercise Files\\carData' + str(today) + '.csv')

        port = 465
        password = 'GregLeeSucks1!'
        sender_email = 'seniorprojectsending@gmail.com'
        receiver_email = 'seniorprojectreceiving@gmail.com'
        subject = str(today) + ' Data'
        body = 'this email contains car data'
        smtp_server = "smtp.gmail.com"


        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject

        message.attach(MIMEText(body, "plain"))

        filename = "carData" + str(today) + ".csv"

        with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        encoders.encode_base64(part)

        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        message.attach(part)
        text = message.as_string()




        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)

        carData['color'] = []
        carData['size'] = []
        carData['time'] = []
        

        



    
    
    if writer is not None:
	    writer.write(frame)

    # show the output frame
    # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    # frameS = cv2.resize(frame, (1248, 702)) THIS ONE IS GOOD
    # frameS = cv2.resize(frame, (1248, 702))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF



	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
    totalFrames += 1


# today = date.today()
# carData.to_csv('C:\\Users\\19732\\Ex_Files_Learning_Python\\Exercise Files\\carData' + str(today) + '.csv')

# port = 465
# password = 'GregLeeSucks1!'
# sender_email = 'seniorprojectsending@gmail.com'
# receiver_email = 'seniorprojectreceiving@gmail.com'
# subject = str(today) + ' Data'
# body = 'this email contains car data'
# smtp_server = "smtp.gmail.com"


# message = MIMEMultipart()
# message["From"] = sender_email
# message["To"] = receiver_email
# message["Subject"] = subject

# message.attach(MIMEText(body, "plain"))

# filename = "carData" + str(today) + ".csv"

# with open(filename, "rb") as attachment:
#         # Add file as application/octet-stream
#         # Email client can usually download this automatically as attachment
#     part = MIMEBase("application", "octet-stream")
#     part.set_payload(attachment.read())

# encoders.encode_base64(part)

# part.add_header(
#     "Content-Disposition",
#     f"attachment; filename= {filename}",
# )

# message.attach(part)
# text = message.as_string()




# context = ssl.create_default_context()

# with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#     server.login(sender_email, password)
#     server.sendmail(sender_email, receiver_email, text)



# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
# if not args.get("input", False):
# 	vs.stop()

# otherwise, release the video file pointer
# else:
# 	vs.release()

# close any open windows
cv2.destroyAllWindows()    



    
#     # If same number of blobs in each frame, see if any blobs are new, or if they're the same ones
#     # if len(prevX) == len(currentX):
#     #     dists = []
#     #     for i in range (0, len(prevX)):
#     #         dists.append(math.sqrt((currentX[i] - prevX[i])**2 + (currentY[i] - prevY[i])**2))

#     #     tooClose = 200
#     #     toBeRemoved = []
#     #     for i in range(0, len(dists)):
#     #         if dists[i] < tooClose:
#     #             toBeRemoved.append(dists[i])

#     #     for item in toBeRemoved:
#     #         dists.remove(item)

#     #     if len(dists) != len(prevX):
#     #         totBlobs = totBlobs + len(prevX) - len(dists)
        
    # Draw the contours on the original frame
    #cv2.drawContours(frame, current_contours_area, -1, (0,255,0), 3)


#     # Counting + recording blobs

#     currentBlobCount = len(current_contours_area)

#     # Showing the blob count on the screen

#     cv2.rectangle(frame, (600, 2), (690, 20), (255, 255, 255), -1)
#     cv2.putText(frame, str(currentBlobCount),(605, 15), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

#     # Showing Cum Blob Count
#     cv2.rectangle(frame, (300, 2), (390, 20), (255, 255, 255), -1)
#     cv2.putText(frame, str(totBlobs),(305, 15), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))




#     frameNo = capture.get(cv2.CAP_PROP_POS_FRAMES)
#     timeElapsed = capture.get(cv2.CAP_PROP_POS_FRAMES) / 50

#     # List the frame number in the original video
#     cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
#     cv2.putText(frame, str(frameNo), (15, 15),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
#     cv2.rectangle(frame, (1250, 2), (1340, 20), (255, 255, 255), -1)
#     cv2.putText(frame, str(timeElapsed) + 's',(1255, 15), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) )
    

# #     #Show the videos
#     cv2.imshow('Frame', frame)
# #     #cv2.imshow('FG Mask', fgMask)



#     # Frankly I'm unclear what this does
#     keyboard = cv2.waitKey(3)
#     if keyboard == 'q' or keyboard == 27:
#         break


# Actual Number of cars = 20


# Print Color of Each Car
        
    # colorCounter = 0

    # for (lower, upper) in colorBorders:
    #     colorCounter = colorCounter + 1

	#     # create NumPy arrays from the boundaries
    #     low = np.array(lower, dtype = "uint8")
    #     high = np.array(upper, dtype = "uint8")
	    
    #     # find the colors within the specified boundaries and apply
	#     # the mask

    #     mask = cv2.inRange(frame, low, high)
    #     output = cv2.bitwise_and(frame, frame, mask = mask)
    #     threshold, outputBW = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #     # Find contours of each individual color
    #     colorContours, useless = cv2.findContours(outputBW, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     # See which color the contours are and print it
    #     for contour in colorContours:
    #         if cv2.contourArea(contour) > 10000:
    #             if colorCounter == 1:
    #                 print("Red")
    #             elif colorCounter == 2:
    #                 print("Blue")
    #             elif colorCounter == 3:
    #                 print("Yellow")
    #             elif colorCounter == 4:
    #                 print ('Grey')
    #             elif colorCounter == 5: 
    #                 print('Black')
    #             elif colorCounter == 6:
    #                 print('White')
    #             elif colorCounter == 7:
    #                 print('Green')
    #             elif colorCounter == 8:
    #                 print('Orange')


    # MY WAY OF FINDING BLOBS (KINDA GOATED)
        # #background subtraction + conversion to grayscale
        # fgMask = backSub.apply(frame)

        # # Gaussian blur ??
        # fgMaskGauss = cv2.GaussianBlur(fgMask, (5,5), 0)

        # # Converting to Black and White
        # (thresh, fgMaskBW) = cv2.threshold(fgMaskGauss, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        # # Getting contours
        # contours, hierarchy = cv2.findContours(fgMaskBW, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contours_area = []

        # # calculate area and filter into new array
        # for con in contours:
        #     area = cv2.contourArea(con)
        #     if area > 2750 and area < 50000:
        #         contours_area.append(con)

        # startXs = []
        # startYs = []
        # endXs = []
        # endYs = []

        # # Calculate the centroids for each contour
        # for c in contours_area:


        #     M = cv2.moments(c)

        #     # From moments, calculate centroids

        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])

        #     startXs.append(cX-10)
        #     endXs.append(cX + 10)
        #     startYs.append(cY+10)
        #     endYs.append(cY-10)

        # # Create a tracker for each object in the frame

        # for i in range(0, len(startXs)):

        #     tracker = dlib.correlation_tracker()
        #     rect = dlib.drectangle(startXs[i], startYs[i], endXs[i], endYs[i])
        #     tracker.start_track(rgb, rect)

        #     # Add tracker to list of trackers
        #     trackers.append(tracker)



# Rendered 22 cars when there were actually 20  

# Rendered 10 cars when there were actually 11 (slomo video helped)

# 