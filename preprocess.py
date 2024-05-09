#Use the following to install mediapipe and the facelandmarker blendshape
#!pip install mediapipe
#!wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2

import cv2

import matplotlib.pyplot as plt
import numpy as np


def contrast_boost(img):
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    return enhanced_img



# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

def landmark_crop(image):

    md_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= image )

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(md_image)
    try:
        landmarks = detection_result.face_landmarks[0]
    except:
        return None
    lips = [landmarks[x] for x in [57, 287, 164, 18]]
    nl = []
    
    for lip in lips:
        nl.append(landmark_pb2.NormalizedLandmark(x=lip.x, y=lip.y, z=lip.z))
        (x, y, z) = image.shape

    # print(nl)
    # print(x, y)
    dim = [x, x, y, y]
    scale = [nl[0].x, nl[1].x, nl[2].y, nl[3].y]
    xy = [int(a*b) for a,b in zip(scale,dim)]
    
    # print(xy)
    X_size_diff = max(0, 48 - (xy[3] - xy[2]+1))
    if X_size_diff > 0:
        xy[3] += (X_size_diff+1)//2
        xy[2] -= X_size_diff//2
    
    Y_size_diff = max(0, 48 - (xy[1] - xy[0]+1))
    if Y_size_diff > 0:
        xy[1] += (Y_size_diff+1)//2
        xy[0] -= Y_size_diff//2
    
    return image[xy[2]:xy[3], xy[0]:xy[1]]

    # STEP 5: Process the detection result. In this case, visualize it.
    #annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    #plt.imshow(frames[(0.07, 0.16)][0])
    #plt.show()
    #plt.imshow(annotated_image)


def canny(img, show = False):
    edges = cv2.Canny(img,200,400)
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
    return edges


def optical_flow(frames, color):
    optical_flow_outputs = {}
    for start,end in frames:
        snippet = frames[(start,end)]
        
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (5, 5),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Take first frame and find corners in it
        old_gray = cv2.cvtColor(snippet[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(snippet[0])

        images = []

        for frame in snippet:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0.dtype = 'float32'
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv2.add(frame, mask)
            images.append(img[:,:,[2,1,0]])

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        optical_flow_outputs[(start, end)] = np.array(images)
    return optical_flow_outputs


# Function to mask the frames

def mask_frames(frames, mask):
    masked_frames = {}
    for start,end in frames:
        snippet = frames[(start,end)]
        masked_snippet = []
        for frame in snippet:
            masked_snippet.append(cv2.bitwise_and(frame, frame, mask = mask))
        masked_frames[(start, end)] = np.array(masked_snippet)
    return masked_frames

#randomcrop

def random_crop(frames, size):
    cropped_frames = {}
    for start,end in frames:
        snippet = frames[(start,end)]
        cropped_snippet = []
        for frame in snippet:
            x, y = frame.shape[0], frame.shape[1]
            x1 = np.random.randint(0, x-size)
            y1 = np.random.randint(0, y-size)
            cropped_snippet.append(frame[x1:x1+size, y1:y1+size])
        cropped_frames[(start, end)] = np.array(cropped_snippet)
    return cropped_frames
