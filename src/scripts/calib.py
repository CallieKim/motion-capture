#!/usr/bin/env python

from pyk4a import PyK4A, Config, CalibrationType
import cv2
import numpy as np

# ---------- 0) GridBoard size ----------
ARUCO_DICT_NAME  = cv2.aruco.DICT_6X6_250
MARKER_LENGTH    = 0.035   # m
MARKER_SEPARATION= 0.0091   # m
BOARD_ROWS       = 4
BOARD_COLS       = 6

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

board = cv2.aruco.GridBoard(
    (BOARD_COLS, BOARD_ROWS),      # (markersX, markersY)
    MARKER_LENGTH,
    MARKER_SEPARATION,
    aruco_dict
)

# ---------- 1) read intrinsics intrinsics ----------
k4a = PyK4A(Config())
k4a.start()

camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
dist_coeffs   = k4a.calibration.get_distortion_coefficients(
                   CalibrationType.COLOR)[:5]

print("Camera matrix:\n", camera_matrix)
print("Distortion coeffs:", dist_coeffs.ravel())

fs = cv2.FileStorage("azure_kinect_intrinsics.yml", cv2.FILE_STORAGE_WRITE)
fs.write("cameraMatrix", camera_matrix)
fs.write("distCoeffs",  dist_coeffs)
fs.release()
print("✅ Saved to azure_kinect_intrinsics.yml")

# ---------- 2) capture a color frame ----------
cap = k4a.get_capture()
if cap.color is None:
    k4a.stop()
    raise RuntimeError("No color frame captured")
frame = cap.color[:, :, :3]        # BGRA → BGR
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ---------- 3) detect markers ----------
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

corners, ids, rejected = detector.detectMarkers(gray)
if ids is None or len(ids) == 0:
    k4a.stop()
    raise RuntimeError("No ArUco markers detected")

# ---------- 4) estimate board pose ----------
valid, rvec_board, tvec_board = cv2.aruco.estimatePoseBoard(
    corners, ids, board, camera_matrix, dist_coeffs, None, None)

if valid <= 0:
    k4a.stop()
    raise RuntimeError("Board pose estimation failed")

frame = frame.copy()  
cv2.aruco.drawDetectedMarkers(frame, corners, ids)
cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                  rvec_board, tvec_board, 0.10)  # 10 cm

cv2.imshow("Pose", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

k4a.stop()

R_b2c, _  = cv2.Rodrigues(rvec_board)
R_c2b     = R_b2c.T
t_cam_in_board = -R_c2b @ tvec_board
rvec_c2b, _ = cv2.Rodrigues(R_c2b)
print("Camera in Board CS  rvec:", rvec_c2b.ravel())
print("Camera in Board CS  tvec:", t_cam_in_board.ravel(), "m")

fs_ext = cv2.FileStorage("azure_kinect_extrinsics.yml", cv2.FILE_STORAGE_WRITE)
fs_ext.write("rvec", rvec_board)
fs_ext.write("tvec", tvec_board)
fs_ext.release()
print("✅ Saved to azure_kinect_extrinsics.yml")