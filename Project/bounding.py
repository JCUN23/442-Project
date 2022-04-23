from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from object_detection_bounding import classify


def drawBox(frame, x, y, w, h):
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	return frame

def write_video(file_path, frames, fps):
	"""
	Writes frames to an mp4 video file
	:param file_path: Path to output video, must end with .mp4
	:param frames: List of PIL.Image objects
	:param fps: Desired frame rate
	"""
	h, w = frames[0].shape[:2]
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

	for frame in frames:
		writer.write(frame)

	writer.release() 
#Runs image classifier, then adds a bounding box on every classified object
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
args = vars(ap.parse_args())
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args["video"] = "images/1080p.mp4"

if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

(major, minor) = cv2.__version__.split(".")[:2] #Get Python Version

# initialize the FPS throughput estimator
fps = None
frame_count = 0

# loop over frames from the video stream
first_frame = True

boundingBoxes = []
trackers = []

width  = vs.get(3)  # float `width`
height = vs.get(4)  # float `height`

frames = []
#video = cv2.VideoWriter('filename.avi', 
                         #cv2.VideoWriter_fourcc(*'MJPG'),
                         #10, frameSize=(width,height),fps=10)
while len(frames) < 300:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	frame_count += 1
	if frame_count == vs.get(cv2.CAP_PROP_FRAME_COUNT):
		frame_count = 0 #Or whatever as long as it is the same as next line
		vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	
	#Classify first frame and initialize tracker and bounding boxes
	if frame_count % 5 == 0:
		trackers.clear()
		boundingBoxes = classify(frame)
		first_frame = False
		num_classified = len(boundingBoxes)
		print(f"classified {num_classified}")
		#Initialize Trackers
		for i in range(num_classified):
			# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
			# function to create our object tracker
			if int(major) == 3 and int(minor) < 3:
				tracker = cv2.Tracker_create(args["tracker"].upper())
			# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
			# approrpiate object tracker constructor:
			else:
				# initialize a dictionary that maps strings to their corresponding
				# OpenCV object tracker implementations
				OPENCV_OBJECT_TRACKERS = {
					"csrt": cv2.TrackerCSRT_create,
					"kcf": cv2.TrackerKCF_create
				}
				# grab the appropriate object tracker using our dictionary of
				# OpenCV object tracker objects
				#tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
				tracker = cv2.TrackerCSRT_create() #hardcode which tracker - needs fix
			# initialize the bounding box coordinates of the object we are going
			# to track

			#initialize bounding boxes
			#boundingBoxes.append(boundingBoxes[i])
			box = boundingBoxes[i]
			tracker.init(frame, box)

			trackers.append(tracker) #Add tracker to array of trackers

			fps = FPS().start() #Start counting FPS

	# check to see if we are currently tracking an object
	if boundingBoxes is not None:
		for tracker in trackers:
			# grab the new bounding box coordinates of the object
			(success, box) = tracker.update(frame)
			# check to see if the tracking was a success
			if success:
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),
					(0, 255, 0), 2)
			else:
				trackers.remove(tracker)
			# update the FPS counter
			fps.update()
			fps.stop()
			# initialize the set of information we'll be displaying on
			# the frame
			info = [
				#("Tracker", args["tracker"]),
				("Success", "Yes" if success else "No"),
				("FPS", "{:.2f}".format(fps.fps())),
			]
			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	# show the output frame
	#video.save(frame)
	frames.append(frame)
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	"""
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)
		fps = FPS().start()
	"""
		
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
#video.release()
write_video("tracking.mp4", frames, 30)
# close all windows
cv2.destroyAllWindows()