from PyQt5.QtGui import QImage, QPixmap
import sys
from PyQt5.QtWidgets import *
import depthai as dai
from pathlib import Path
from HoughTan import *
import HouUITX2

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

topLeft = dai.Point2f(0, 0)
bottomRight = dai.Point2f(0, 0)
topLeft2 = dai.Point2f(100, 100)
bottomRight2 = dai.Point2f(100, 100)
corp1 = True
corpover = False
clearflag = True
index = 0

# fmt: off
labelMap = [
    "Rebar"
]

class test_ui(QMainWindow, HouUITX2.Ui_MainWindow):
    def __init__(self):
        super().__init__()  
        self.setupUi(self)
        self.setWindowTitle("Steel_Measurement")
        self.pushButton_3.setEnabled(False)
        self.pushButton_5.clicked.connect(sys.exit)
        self.canny_params = [(50, 100), (100, 200), (200, 400)][self.comboBox4.currentIndex()]

    def SetFixed(self):
        global FixedFlag, PauseFlag
        PauseFlag = True
        FixedFlag = True
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(True)
    def Restore(self):
        global FixedFlag, PauseFlag
        PauseFlag = False
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(False)
    def choose_pic(self):
        global index
        index = self.comboBox.currentIndex()
    def mouse_click_event(self, event):
        global refPt, click_roi, corp1, refPt2, click_roi2, corpover, clearflag
        x, y = event.x(), event.y()
        x = x / 0.4
        y = y / 0.8
        clearflag = False
        if corp1 == True:
            refPt = [(x, y)]
            click_roi = ([(x-5, y-5), (x-5, y-5)])
            corp1 = False
        elif corp1 == False:
            refPt2 = [(x, y)]
            click_roi2 = ([(x - 5, y - 5), (x - 5, y - 5)])
            corp1 = True
            corpover = True
    def mouse_click_event2(self, event):
        global corpover, clearflag, topLeft2, bottomRight2
        corpover = False
        clearflag = True
        topLeft2 = dai.Point2f(0, 0)
        bottomRight2 = dai.Point2f(0, 0)


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
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

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

    # try:
    #     calibJsonFile = str(
    #         (Path(__file__).parent / Path('calib_18443010C1B8840E00.json')).resolve().absolute())
    #
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('calibJsonFile', nargs='?', help="Path to calibration file in json", default=calibJsonFile)
    #     args = parser.parse_args()
    #     calibData = dai.CalibrationHandler(args.calibJsonFile)
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
    stereo.initialConfig.set(configs)

    # Network specific settings
    spatialDetectionNetwork.setBlob(model)
    spatialDetectionNetwork.setConfidenceThreshold(0.20)

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

    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft, bottomRight)
    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    config2 = dai.SpatialLocationCalculatorConfigData()
    config2.depthThresholds.lowerThreshold = 100
    config2.depthThresholds.upperThreshold = 10000
    config2.roi = dai.Rect(topLeft2, bottomRight2)
    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config2)

    # Linking
    camRgb.isp.link(rgbOut.input)
    camRgb.preview.link(spatialDetectionNetwork.input)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    stereo.disparity.link(disparityOut.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    spatialDetectionNetwork.passthroughDepth.link(spatialLocationCalculator.inputDepth)
    spatialDetectionNetwork.out.link(xoutNN.input)

    spatialLocationCalculator.out.link(xoutSpatialData.input)

    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    return pipeline, stereo.initialConfig.getMaxDisparity(), camRgb

def check_input(roi, frame, DELTA=5):
    """
    Check if input is ROI or point. If point, convert to ROI
    """
    if len(roi) == 2:
        if len(roi[0]) == 2:
            roi = np.array(roi) + [[-DELTA, -DELTA], [DELTA, DELTA]]
        else:
            roi = np.array([roi, roi]) + [[-DELTA, -DELTA], [DELTA, DELTA]]
    elif len(roi) == 4:
        roi = np.array(roi) + [[-DELTA, -DELTA], [DELTA, DELTA]]

    roi.clip([DELTA, DELTA], [frame.shape[1] - DELTA, frame.shape[0] - DELTA])

    return roi / frame.shape[1::-1]


def click_and_crop(event, x, y, flags, param):
    global refPt, click_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        refPt = np.array(refPt)
        click_roi = np.array([np.min(refPt, axis=0), np.max(refPt, axis=0)])

def click_and_crop2(event, x, y, flags, param):
    global refPt2, click_roi2
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt2 = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt2.append((x, y))
        refPt2 = np.array(refPt2)
        click_roi2 = np.array([np.min(refPt2, axis=0), np.max(refPt2, axis=0)])

def run():
    global dm, FixedFlag, PauseFlag, refPt, click_roi, refPt2, click_roi2
    newConfig = False
    # Connect to device and start pipeline
    with dai.Device() as device:
        S3D=[]
        E3D=[]
        pipeline, maxDisparity, camRgb = create_pipeline(device)
        device.startPipeline(pipeline)
        device.setIrLaserDotProjectorBrightness(500)
        device.setIrFloodLightBrightness(500)  # in mA, 0..1500
        frameRgb = None
        frameDisp = None
        depthDatas = []
        detections = []
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        imageQueue = device.getOutputQueue("rgb")
        dispQueue = device.getOutputQueue("disp")
        spatialDataQueue = device.getOutputQueue("spatialData")
        detectQueue = device.getOutputQueue(name="detections")

        def frameNorm(frame, bbox):
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

        def drawSpatialLocations(frame, spatialLocations):
            for depthData in spatialLocations:
                roi = depthData.config.roi
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                if clearflag == False:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 4)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

                    drawText(
                        frame,
                        f"X: {int(depthData.spatialCoordinates.x)} mm",
                        (xmin + 10, ymin + 20),
                    )
                    drawText(
                        frame,
                        f"Y: {int(depthData.spatialCoordinates.y)} mm",
                        (xmin + 10, ymin + 35),
                    )
                    drawText(
                        frame,
                        f"Z: {int(depthData.spatialCoordinates.z)} mm",
                        (xmin + 10, ymin + 50),
                    )

        def line_equation(x1, y1, x2, y2):
            if x2 - x1 != 0:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
            else:
                m = np.inf
                b = x1
            return m, b

        def drawDetection(frame, detections, dispdata):
            i = 0
            S3D=[]
            E3D=[]
            for detection in detections:
                bbox = frameNorm(frame,(detection.xmin, detection.ymin, detection.xmax, detection.ymax),)
                roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                if roi is None:
                    print("ROI is None")
                    continue
                if roi.size == 0:
                    print("ROI is empty")
                    continue

                try:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    print(f"cv2.cvtColor error: {e}")
                    print(f"ROI shape: {roi.shape}")
                    continue

                edges = cv2.Canny(gray, *win.canny_params)
                angle = get_main_axis_angle_pca(edges)
                # angle = 30
                if angle is not None:
                    lines = hough_transformP2_vectorized(edges, angle, 5, threshold=220, min_line_length=100,
                                             merge_threshold_rho=10, merge_threshold_theta=np.pi / 90 * 2)
                    if lines is not None:
                        for line in lines:
                            (x1, y1), (x2, y2) = line
                            cv2.line(frame, (bbox[0] + x1, bbox[1] + y1), (bbox[0] + x2, bbox[1] + y2), (0, 0, 255), 1)
                        if len(lines) >= 2:
                            middle_slope, middle_intercept, s, ii = calculate_middle_line(lines)
                            mid_start, mid_end, line_start, line_end = draw_middle_line(edges, middle_slope, middle_intercept)
                            mid_points = [mid_start, mid_end]

                            cv2.line(frame, (int(bbox[0] + mid_start[0]), int(bbox[1] + mid_start[1])),
                                     (int(bbox[0] + mid_end[0]), int(bbox[1] + mid_end[1])),
                                     (255, 0, 0),
                                     1)  # 红色
                            depth1 = depth2 = detection.spatialCoordinates.z

                            start_3d = calculate_3d_coordinates((bbox[0] + mid_start[0]), (bbox[1] + mid_start[1]), depth1, frame,
                                                                focal_length=(795.8499755859375+787.8486328125)/2)
                            end_3d = calculate_3d_coordinates((bbox[0] + mid_end[0]), (bbox[1] + mid_end[1]), depth2, frame,
                                                              focal_length=(795.8499755859375+787.8486328125)/2)
                            S3D.append(start_3d)
                            E3D.append(end_3d)
                        else:
                            start_3d = (0, 0, 0)
                            end_3d = (0, 0, 0)
                            S3D.append(start_3d)
                            E3D.append(end_3d)


            for detection in detections:
                bbox = frameNorm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 4)
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                drawText(frame, f"ID:{i + 1}", (bbox[0] + 10, bbox[1] + 30), )
                i += 1
            return S3D, E3D

        def Add_Combobox(detections):
            global dm
            for d in range(len(detections)):
                win.comboBox2.setItemText(d, f'ID:{d+1}')
                win.comboBox3.setItemText(d, f'ID:{d+1}')
            dm = len(detections)

        def Spatial_Measurement(Start_3D, End_3D):
            index2 = win.comboBox2.currentIndex()
            index3 = win.comboBox3.currentIndex()
            X = []
            Y = []
            Z = []
            if index2 == index3:
                win.label2.setText("ERROR")
            else:
                L = calculate_3d(Start_3D[index2], End_3D[index2], Start_3D[index3], End_3D[index3])
                L = calculate_spacing(L)
                win.label2.setText("{:.2f}".format(L))

        def draw_middle_line(image, middle_slope, middle_intercept):
            height, width = image.shape[:2]
            is_vertical = abs(middle_slope) > 1
            if is_vertical:
                y1, y2 = 0, height - 1
                x1 = int((y1 - middle_intercept) / middle_slope)
                x2 = int((y2 - middle_intercept) / middle_slope)
                x1 = min(max(x1, 0), width - 1)
                x2 = min(max(x2, 0), width - 1)
                line_start = (x1, y1)
                line_end = (x2, y2)
            else:
                x1, x2 = 0, width - 1
                y1 = int(middle_slope * x1 + middle_intercept)
                y2 = int(middle_slope * x2 + middle_intercept)
                y1 = min(max(y1, 0), height - 1)
                y2 = min(max(y2, 0), height - 1)
                line_start = (x1, y1)
                line_end = (x2, y2)
            line_length = np.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])
            mid_length = line_length / 3
            mid_start = (
                line_start[0] + (line_end[0] - line_start[0]) * (1 / 3),
                line_start[1] + (line_end[1] - line_start[1]) * (1 / 3)
            )
            mid_end = (
                line_start[0] + (line_end[0] - line_start[0]) * (2 / 3),
                line_start[1] + (line_end[1] - line_start[1]) * (2 / 3)
            )

            return mid_start, mid_end, line_start, line_end

        def calculate_middle_line(lines):
            if len(lines) < 2:
                raise ValueError("ERROR")
            slopes = []
            intercepts = []
            for line in lines:
                (x1, y1), (x2, y2) = line

                m, b = line_equation(x1, y1, x2, y2)
                slopes.append(m)
                intercepts.append(b)
            middle_slope = np.mean(slopes)
            middle_intercept = np.mean(intercepts)
            return middle_slope, middle_intercept, slopes, intercepts

        def drawL(frame, spatialLocations1, spatialLocations2):
            for depthData1 in spatialLocations1:
                roi = depthData1.config.roi
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                x1min = int(roi.topLeft().x)
                y1min = int(roi.topLeft().y)
                x1max = int(roi.bottomRight().x)
                y1max = int(roi.bottomRight().y)
                a1 = depthData1.spatialCoordinates.x
                b1 = depthData1.spatialCoordinates.y
                c1 = depthData1.spatialCoordinates.z
            for depthData2 in spatialLocations2:
                roi = depthData2.config.roi
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                x2min = int(roi.topLeft().x)
                y2min = int(roi.topLeft().y)
                x2max = int(roi.bottomRight().x)
                y2max = int(roi.bottomRight().y)
                a2 = depthData2.spatialCoordinates.x
                b2 = depthData2.spatialCoordinates.y
                c2 = depthData2.spatialCoordinates.z
            if clearflag == False:
                cv2.line(frame, (x1max, y1max), (x2max, y2max),
                                    (0, 0, 255), 2)
                L = (((a1 - a2) ** 2 + (b1 - b2) ** 2 + (c1 - c2 ) ** 2) ** 0.5)
                drawText(
                    frame,
                    f"L: {int(L)} mm",
                    (150, 50),
                )
        while not device.isClosed():
            win.pushButton_2.clicked.connect(win.SetFixed)
            win.pushButton_3.clicked.connect(win.Restore)
            index = win.comboBox.currentIndex()
            if PauseFlag is False:
                imageData = imageQueue.tryGet()
                dispData = dispQueue.tryGet()
                detData = detectQueue.tryGet()
                spatialData = spatialDataQueue.tryGet()

                if spatialData is not None:
                    depthDatas = spatialData.getSpatialLocations()
                if len(depthDatas) != 0:
                    X1 = [depthDatas[0]]
                    X2 = [depthDatas[1]]

                if detData is not None:
                    detections = detData.detections
                    Add_Combobox(detections)

                if imageData is not None:
                    frameRgb = imageData.getCvFrame()
                    if corpover == False:
                        win.label.mousePressEvent = win.mouse_click_event
                    if corpover == True:
                        win.label.mousePressEvent = win.mouse_click_event2

                if dispData is not None:
                    frameDisp = dispData.getFrame()
                    frameDisp2 = (frameDisp * (255 / maxDisparity)).astype(np.uint8)
                    frameDisp2 = cv2.applyColorMap(frameDisp2, cv2.COLORMAP_JET)
                    frameDisp2 = np.ascontiguousarray(frameDisp2)

                if index == 0:
                    if frameRgb is not None:
                        VTRGB = frameRgb
                        if click_roi is not None:
                            [topLeft.x, topLeft.y], [bottomRight.x, bottomRight.y] = check_input(click_roi, VTRGB)
                            click_roi = None
                            newConfig = True

                        if click_roi2 is not None:
                            [topLeft2.x, topLeft2.y], [bottomRight2.x, bottomRight2.y] = check_input(click_roi2,
                                                                                                     VTRGB)
                            click_roi2 = None
                            newConfig = True
                        VTRGB = cv2.resize(VTRGB, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)
                        if len(depthDatas) != 0:
                            drawSpatialLocations(VTRGB, X1)
                            drawSpatialLocations(VTRGB, X2)
                            if corpover == True:
                                drawL(VTRGB, X1, X2)
                        VTRGB = cv2.cvtColor(VTRGB, cv2.COLOR_BGR2RGB)
                        VRGB_image = QImage(VTRGB.data, VTRGB.shape[1], VTRGB.shape[0], QImage.Format_RGB888)
                        win.label.setPixmap(QPixmap.fromImage(VRGB_image))

                elif index == 1:
                    VTDisp = frameDisp2
                    VTDisp = cv2.resize(VTDisp, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)
                    VTDisp_image = QImage(VTDisp.data, VTDisp.shape[1], VTDisp.shape[0], QImage.Format_BGR888)
                    win.label.setPixmap(QPixmap.fromImage(VTDisp_image))
                elif index == 2:
                    grayFrame = cv2.cvtColor(frameRgb, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(grayFrame, *win.canny_params)
                    edge_image = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_RGB888)
                    win.label.setPixmap(QPixmap.fromImage(edge_image))
            if FixedFlag == True:
                S3D, E3D=drawDetection(frameRgb, detections, frameDisp)
                VTRGB = cv2.cvtColor(frameRgb, cv2.COLOR_BGR2RGB)
                VTRGB = cv2.resize(VTRGB, None, fx=0.4, fy=0.8, interpolation=cv2.INTER_AREA)
                if len(depthDatas) != 0:
                    drawSpatialLocations(VTRGB, X1)
                    drawSpatialLocations(VTRGB, X2)
                    if corpover == True:
                        drawL(VTRGB, X1, X2)
                VRGB_image = QImage(VTRGB.data, VTRGB.shape[1], VTRGB.shape[0], QImage.Format_RGB888)
                win.label.setPixmap(QPixmap.fromImage(VRGB_image))
                FixedFlag = False

            if len(detections) != 0:
                win.pushButton_4.clicked.connect(lambda: Spatial_Measurement(S3D, E3D))

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if newConfig:
                config = dai.SpatialLocationCalculatorConfigData()
                config.depthThresholds.lowerThreshold = 100
                config.depthThresholds.upperThreshold = 10000
                config.roi = dai.Rect(topLeft, bottomRight)
                config.calculationAlgorithm = (
                    dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                )
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)

                config2 = dai.SpatialLocationCalculatorConfigData()
                config2.depthThresholds.lowerThreshold = 100
                config2.depthThresholds.upperThreshold = 10000
                config2.roi = dai.Rect(topLeft2, bottomRight2)
                config2.calculationAlgorithm = (
                    dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                )
                cfg.addROI(config2)
                spatialCalcConfigInQueue.send(cfg)
                newConfig = False
    sys.exit(app.exec_())


if __name__ == "__main__":
    FixedFlag = False
    PauseFlag = False
    app = QApplication(sys.argv)
    win = test_ui()
    win.show()
    refPt = None
    click_roi = None
    refPt2 = None
    click_roi2 = None
    run()
    sys.exit(app.exec_())