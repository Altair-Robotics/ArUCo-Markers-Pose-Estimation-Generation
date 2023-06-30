# program to capture single image from webcam in python

# importing OpenCV library
import cv2
import realsense2_camera as rs
import numpy as np

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cap = cv2.VideoCapture(0)
counter = 0
folder = "images_for_cal"
k = 0

### For realsense ###
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
### For realsense ###


# in a 8x8 chessboard, we detect the inner corners, so 7x7. 
chessboardSize = (7,7)

# criteria for detecting chesscblocks
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


while True:

    succes, img = cap.read()

    ### For realsense ###
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if  not color_frame:
        continue
    
    color_image = np.asanyarray(color_frame.get_data())
    ### For realsense ###
    
    cv2.imshow('img',img)
    k = cv2.waitKey(1)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
    # print(ret)
    
    if ret:	# if ret is True, opencv has successfully found the chessboard corners
        temp_img = img.copy()
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(temp_img, chessboardSize, corners2, ret)
        cv2.imshow('Detected image', temp_img)

    if k == 27: 
        break	# ends if ESC key is pressed 
    
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(f'{folder}/Image_{counter}.png', img)
        print(f"{counter}image saved!")
        counter += 1
        cv2.imshow('saved image',img)   

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()