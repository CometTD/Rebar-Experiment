from PyQt5.QtGui import QImage, QPixmap
import sys
import Test2
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse

# nn part
# numClasses = 1
numClasses = 80

blob = Path(__file__).parent.joinpath("model.blob")
model = dai.OpenVINO.Blob(blob)
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

output_name, output_tenser = next(iter(model.networkOutputs.items()))
if "yolov6" in output_name:
    numClasses = output_tenser.dims[2] - 5
else:
    numClasses = output_tenser.dims[2] // 3 - 5

# fmt: off
labelMap = [
    "Rebar"
]
# labelMap = [
#     "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
#     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
#     "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
#     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
#     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
#     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
#     "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
#     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
#     "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
#     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
#     "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
#     "teddy bear",     "hair drier", "toothbrush"
# ]

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 40
depthWeight = 60
ViewModeFlag = True

class test_ui(QMainWindow, Test2.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Image Viewer")
        self.sl = self.horizontalSlider
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setSingleStep(1)
        self.sl.setValue(40)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setHidden(True)
        self.pushButton_2.setEnabled(False)

    def choose_pic(self):
        global index
        index = self.comboBox.currentIndex()
    def ViewMode(self):
        global ViewModeFlag
        ViewModeFlag = True
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(True)
    def MeasureMode(self):
        global ViewModeFlag
        ViewModeFlag = False
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(False)

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)
    depthWeight = 100 - rgbWeight


def create_pipeline(device):
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    # Create pipeline
    pipeline = dai.Pipeline()
            
    # Define sources and outputs  
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)

    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

    rgbOut.setStreamName("rgb")
    disparityOut.setStreamName("disp")
    xoutNN.setStreamName("detections")

    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(30)
    camRgb.setIspScale(2, 3)
    camRgb.setPreviewSize(W, H)
    camRgb.setInterleaved(False)
    camRgb.setPreviewKeepAspectRatio(False)

    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    # try:
    #     calibJsonFile = str(
    #         (Path(__file__).parent / Path('D:\depth_align_and_nn\calib_18443010C1B8840E00.json')).resolve().absolute())
    #
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('calibJsonFile', nargs='?', help="Path to calibration file in json", default=calibJsonFile)
    #     args = parser.parse_args()
    #
    #     calibData = dai.CalibrationHandler(args.calibJsonFile)
    #     # calibData = device.readCalibration2()
    #     lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    #     # if lensPosition:
    #     #     camRgb.initialControl.setManualFocus(lensPosition)
    # except:
    #     raise
    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(30)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(30)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    configs = stereo.initialConfig.get()
    # configs.postProcessing.speckleFilter.enable = False
    # configs.postProcessing.speckleFilter.speckleRange = 10
    # configs.postProcessing.temporalFilter.enable = False
    # configs.postProcessing.spatialFilter.enable = False
    # configs.postProcessing.spatialFilter.holeFillingRadius = 1
    # configs.postProcessing.spatialFilter.numIterations = 1
    configs.postProcessing.thresholdFilter.minRange = 400
    configs.postProcessing.thresholdFilter.maxRange = 15000
    # configs.postProcessing.decimationFilter.decimationFactor = 1
    # stereo.initialConfig.set(configs)

    # Network specific settings
    spatialDetectionNetwork.setBlob(model)
    spatialDetectionNetwork.setConfidenceThreshold(0.2)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(numClasses)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([])
    spatialDetectionNetwork.setAnchorMasks({})
    spatialDetectionNetwork.setIouThreshold(0.5)



    # spatial specific parameters
    spatialDetectionNetwork.setBoundingBoxScaleFactor(1)
    spatialDetectionNetwork.setDepthLowerThreshold(10)
    spatialDetectionNetwork.setDepthUpperThreshold(10000)

    # Linking
    camRgb.isp.link(rgbOut.input)
    camRgb.preview.link(spatialDetectionNetwork.input)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    stereo.disparity.link(disparityOut.input)
    # stereo.depth.link(spatialLocationCalculator.inputDepth)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)


    spatialDetectionNetwork.out.link(xoutNN.input)

    # spatialLocationCalculator.passthroughDepth.link(spatialLocationCalculator.inputDepth)

    return pipeline, stereo.initialConfig.getMaxDisparity()


def check_input(roi, frame, DELTA=5):
    """
    Check if input is ROI or point. If point, convert to ROI
    """
    # Limit the point so ROI won't be outside the frame
    if len(roi) == 2:
        if len(roi[0]) == 2:
            roi = np.array(roi) + [[-DELTA, -DELTA], [DELTA, DELTA]]
        else:
            roi = np.array([roi, roi]) + [[-DELTA, -DELTA], [DELTA, DELTA]]
    elif len(roi) == 4:
        roi = np.array(roi) + [[-DELTA, -DELTA], [DELTA, DELTA]]

    roi.clip([DELTA, DELTA], [frame.shape[1] - DELTA, frame.shape[0] - DELTA])

    return roi / frame.shape[1::-1]


def run():
    global dm
    # Connect to device and start pipeline
    with dai.Device() as device:
        pipeline,  maxDisparity = create_pipeline(device)
        device.startPipeline(pipeline)
        device.setIrLaserDotProjectorBrightness(600)
        device.setIrFloodLightBrightness(600)  # in mA, 0..1500
        frameRgb = None
        frameDisp = None
        depthDatas = []
        detections = []
        stepSize = 0.01
        newConfig = False
        dm = 0

        # Configure windows; trackbar adjusts blending ratio of rgb/depth
        rgbWindowName = "rgb"
        depthWindowName = "depth"
        blendedWindowName = "rgb-depth"
        # cv2.namedWindow(rgbWindowName)
        # cv2.namedWindow(depthWindowName)
        # cv2.namedWindow(blendedWindowName)
        # cv2.createTrackbar(
        #     "RGB Weight %",
        #     blendedWindowName,
        #     int(rgbWeight),
        #     100,
        #     updateBlendWeights,
        # )

        print("Use WASD keys to move ROI!")

        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        imageQueue = device.getOutputQueue("rgb")
        dispQueue = device.getOutputQueue("disp")
        spatialDataQueue = device.getOutputQueue("spatialData")
        detectQueue = device.getOutputQueue(name="detections")

        def frameNorm(frame, bbox):
            """
            nn data, being the bounding box locations, are in <0..1> range
            - they need to be normalized with frame width/height
            """
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def drawText(frame, text, org, color=(255, 255, 255)):
            cv2.putText(
                frame,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
        def drawTextID(frame, text, org, color=(255, 255, 255)):
            cv2.putText(
                frame,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 0),
                20,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5, cv2.LINE_AA
            )

        def drawDetection(frame, detections):
            # Convert normalized bounding box coordinates to pixel values and calculate side lengths in pixels
            pixel_side_lengths = [
                min(frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))[2] -
                    frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))[0],
                    frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))[3] -
                    frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))[1])
                for detection in detections
            ]
            if pixel_side_lengths:
                threshold = np.mean(pixel_side_lengths) + np.std(pixel_side_lengths)
            else:
                threshold = 0  # Default to zero if no detections to avoid errors

            i = 0
            for detection in detections:
                # Convert normalized bounding box coordinates to pixel values
                bbox = frameNorm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )

                # Draw label, ID, and bounding box
                drawText(
                    frame,
                    labelMap[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                )
                drawTextID(
                    frame,
                    f"ID:{i + 1}",
                    (bbox[0] + 10, bbox[1] + 95),
                )
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 4
                )
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1
                )

                # Display spatial coordinates if available
                # if hasattr(detection, "spatialCoordinates"):
                #     drawText(
                #         frame,
                #         f"X: {int(detection.spatialCoordinates.x)} mm",
                #         (bbox[0] + 10, bbox[1] + 50),
                #     )
                #     drawText(
                #         frame,
                #         f"Y: {int(detection.spatialCoordinates.y)} mm",
                #         (bbox[0] + 10, bbox[1] + 65),
                #     )
                #     drawText(
                #         frame,
                #         f"Z: {int(detection.spatialCoordinates.z)} mm",
                #         (bbox[0] + 10, bbox[1] + 80),
                #     )

                # Calculate the shortest side length of the bounding box in pixels
                shortest_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

                # Display "倾斜" if the shortest side exceeds the threshold
                if shortest_side > threshold:
                    drawText(
                        frame,
                        "tilted",
                        (bbox[0] + 10, bbox[1] + 35),  # Positioning below the bounding box
                        color=(0, 0, 255)  # Display in red
                    )

                i += 1

        # def drawDetection(frame, detections):
        #     i = 0
        #     for detection in detections:
        #         bbox = frameNorm(
        #             frame,
        #             (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
        #         )
        #         drawText(
        #             frame,
        #             labelMap[detection.label],
        #             (bbox[0] + 10, bbox[1] + 20),
        #         )
        #         # drawText(
        #         #     frame,
        #         #     f"{detection.confidence:.2%}",
        #         #     (bbox[0] + 10, bbox[1] + 35),
        #         # )
        #         drawText(
        #             frame,
        #             f"ID:{i + 1}",
        #             (bbox[0] + 10, bbox[1] + 95),
        #         )
        #         cv2.rectangle(
        #             frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 4
        #         )
        #         cv2.rectangle(
        #             frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1
        #         )
        #         if hasattr(detection, "boundingBoxMapping"):
        #             drawText(
        #                 frame,
        #                 f"X: {int(detection.spatialCoordinates.x)} mm",
        #                 (bbox[0] + 10, bbox[1] + 50),
        #             )
        #             drawText(
        #                 frame,
        #                 f"Y: {int(detection.spatialCoordinates.y)} mm",
        #                 (bbox[0] + 10, bbox[1] + 65),
        #             )
        #             drawText(
        #                 frame,
        #                 f"Z: {int(detection.spatialCoordinates.z)} mm",
        #                 (bbox[0] + 10, bbox[1] + 80),
        #             )
        #             i += 1

        def Measurement(detections):
            index2 = win.comboBox2.currentIndex()
            index3 = win.comboBox3.currentIndex()
            X = []
            Y = []
            Z = []
            for detection in detections:
                X.append(detection.spatialCoordinates.x)
                Y.append(detection.spatialCoordinates.y)
                Z.append(detection.spatialCoordinates.z)
            if index2 == index3:
                win.label2.setText("ERROR")
            else:
                if len(X) >= index3+1 and len(X) >= index2+1:
                    L = (((X[index2] - X[index3]) ** 2 + (Y[index2] - Y[index3]) ** 2 + (Z[index2] - Z[index3]) ** 2)
                         ** 0.5)
                    # RL = L - (0.1 * L + 5)
                    RL = L
                    win.label2.setText("{:.4f}".format(RL))
                else:
                    win.label2.setText("ERROR")

        def Add_Combobox(detections):
            global dm
            # if len(detections) != dm:
            #     win.comboBox2.clear()
            #     win.comboBox3.clear()
            #     for d in range(len(detections)):
            #         win.comboBox2.addItem("")
            #         win.comboBox3.addItem("")
            for d in range(len(detections)):
                win.comboBox2.setItemText(d, f'ID:{d+1}')
                win.comboBox3.setItemText(d, f'ID:{d+1}')
            dm = len(detections)

        while not device.isClosed():
            win.pushButton_2.clicked.connect(win.ViewMode)
            win.pushButton_3.clicked.connect(win.MeasureMode)
            rgbWeight = win.sl.value()
            updateBlendWeights(rgbWeight)
            if ViewModeFlag == True:
                imageData = imageQueue.tryGet()
                dispData = dispQueue.tryGet()
                detData = detectQueue.tryGet()

            if detData is not None:
                detections = detData.detections
                Add_Combobox(detections)

            if len(detections) != 0:
                win.pushButton_4.clicked.connect(lambda: Measurement(detections))

            if len(depthDatas) != 0:
                X1 = [depthDatas[0]]
                X2 = [depthDatas[1]]
                # drawXYZ1(X1)
                # drawXYZ2(X2)
                # drawL(X1, X2)

            if imageData is not None:
                frameRgb = imageData.getCvFrame()
                frameRgbModeView = frameRgb
                drawDetection(frameRgb, detections)
                # if len(depthDatas) != 0:
                    # drawSpatialLocations(frameRgb, X1)
                    # drawSpatialLocations(frameRgb, X2)
                    # drawL(frameRgb, X1, X2)
                # drawSpatialLocations2(depthDatas)

                # cv2.imshow(rgbWindowName, frameRgb)

            if dispData is not None:
                frameDisp = dispData.getFrame()
                frameDisp = (frameDisp * (255 / maxDisparity)).astype(np.uint8)
                frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)
                frameDisp = np.ascontiguousarray(frameDisp)
                frameDispModeView = frameDisp

            # Blend when both received
            if frameRgb is not None and frameDisp is not None:
                # Need to have both frames in BGR format before blending
                if len(frameDisp.shape) < 3:
                    frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
                blended = cv2.addWeighted(
                    frameRgb, rgbWeight / 100, frameDisp, depthWeight / 100, 0
                )
                if index == 0:
                    VTRGB = frameRgbModeView
                    VTRGB = cv2.resize(VTRGB, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)
                    # VTRGB = cv2.normalize(VTRGB, None, 0, 255, cv2.NORM_MINMAX)
                    # VTRGB = VTRGB.astype('uint8')
                    # TRGB = cv2.applyColorMap(TRGB, cv2.COLORMAP_JET)
                    VTRGB = cv2.cvtColor(VTRGB, cv2.COLOR_BGR2RGB)
                    VRGB_image = QImage(VTRGB.data, VTRGB.shape[1], VTRGB.shape[0], QImage.Format_RGB888)
                    win.label.setPixmap(QPixmap.fromImage(VRGB_image))
                    win.sl.setVisible(False)

                elif index == 1:
                    VTDisp = frameDispModeView
                    VTDisp = cv2.resize(VTDisp, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)
                    VTDisp_image = QImage(VTDisp.data, VTDisp.shape[1], VTDisp.shape[0], QImage.Format_BGR888)
                    win.label.setPixmap(QPixmap.fromImage(VTDisp_image))
                    win.sl.setVisible(False)
                elif index == 2:
                    Tblend = blended
                    Tblended_image = QImage(Tblend.data, Tblend.shape[1], Tblend.shape[0], QImage.Format_BGR888)
                    win.label.setPixmap(QPixmap.fromImage(Tblended_image))
                    win.sl.setVisible(True)

                # cv2.imshow(blendedWindowName, blended)
                # frameRgb = None
                # frameDisp = None
                # depthDatas = []

            index = win.comboBox.currentIndex()

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = test_ui()
    win.show()
    run()

