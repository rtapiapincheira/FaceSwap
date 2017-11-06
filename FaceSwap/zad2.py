# coding=utf-8

import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils
import os
import time

print "Press T to draw the keypoints and the 3D model"
print "Press R to start recording to a video file"


# Gets current time
def current_time_millis():
    return int(round(time.time() * 1000))


# List files in the specified folder
def list_files(folder):
    result = []
    for f in os.listdir(folder):
        fs = f.lower()
        if fs.endswith(".png") or fs.endswith(".jpg") or fs.endswith(".gif"):
            result.append(folder + "/" + f)
    return result


# Load images and check at least 1 exist
all_images = list_files("../images")
if len(all_images) == 0:
    print "Debes colocar 1 o más imáges en la carpeta /images"
    exit(1)

# Time before rotating the images
time_before_change = 30000

# The smaller this value gets the faster the detection will work if it is too small, the user's face might not be
# detected
maxImageSizeForDetection = 360

detector = dlib.get_frontal_face_detector()

# You need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file
# is http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

# Loading the keypoint detection model, the image and the 3D model
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("../candide.npz")

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]

current_image = 0

# This will force the image to be processed as soon as entering the loop
last_processed = 0

cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)

while True:

    now = current_time_millis()

    # Logic for rotating the images
    if now - last_processed >= time_before_change:
        current_image += 1
        current_image %= len(all_images)

        print 'Processing model image: ', all_images[current_image]

        # Create model for fist image
        textureImg = cv2.imread(all_images[current_image])
        textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector,
                                                   predictor)
        renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

        last_processed = now

    cameraImg = cap.read()[1]
    cameraImg = cv2.flip(cameraImg, 1)

    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    if shapes2D is not None:
        for shape2D in shapes2D:
            # 3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            # 3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual,
                                                            projectionModel.jacobian, (
                                                            [mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]],
                                                            shape2D[:, idxs2D]), verbose=0)

            # rendering the model to an image
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            # blending of the rendered face with the image
            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
            cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)

            # drawing of the mesh and keypoints
            if drawOverlay:
                drawPoints(cameraImg, shape2D.T)
                drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams,
                                   lockedTranslation)

    if writer is not None:
        writer.write(cameraImg)

    display_width = 1440
    display_height = 900

    # Comentar la siguiente linea para evitar "agrandar" la imagen para que se ajuste a la pantalla
    cameraImg = cv2.resize(cameraImg, (display_width, display_height))

    cv2.imshow("test", cameraImg)

    # Comentar para quitar el fullscreen
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('t'):
        drawOverlay = not drawOverlay
    if key == ord('r'):
        if writer is None:
            print "Starting video writer"
            writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25,
                                     (cameraImg.shape[1], cameraImg.shape[0]))

            if writer.isOpened():
                print "Writer succesfully opened"
            else:
                writer = None
                print "Writer opening failed"
        else:
            print "Stopping video writer"
            writer.release()
            writer = None
