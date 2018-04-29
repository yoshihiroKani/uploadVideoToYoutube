# -*- coding:utf-8 -*-
import cv2
import numpy as np

fname = "ero2"

# 入力画像を読み込み
img = cv2.imread(fname + ".jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# カーネル（縦方向の輪郭検出用）
kernel = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
prewitt = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
laplacian = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
ori = np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])

dst = cv2.filter2D(gray, cv2.CV_64F, ori)
# 結果を出力
cv2.imwrite(fname + "_ori.jpg", dst)
# 結果を出力 負の勾配も含む
dst_abs = np.abs(dst)
cv2.imwrite(fname + "_ori_abs.jpg", dst_abs)


dst = cv2.filter2D(gray, cv2.CV_64F, kernel)
# 結果を出力
cv2.imwrite(fname + "_k.jpg", dst)
# 結果を出力 負の勾配も含む
dst_abs = np.abs(dst)
cv2.imwrite(fname + "_k_abs.jpg", dst_abs)

dst = cv2.filter2D(gray, cv2.CV_64F, prewitt)
# 結果を出力
cv2.imwrite(fname + "_prewitt.jpg", dst)
# 結果を出力 負の勾配も含む
dst_abs = np.abs(dst)
cv2.imwrite(fname + "_prewitt_abs.jpg", dst_abs)

dst = cv2.filter2D(gray, cv2.CV_64F, sobel)
# 結果を出力
cv2.imwrite(fname + "_sobel.jpg", dst)
# 結果を出力 負の勾配も含む
dst_abs = np.abs(dst)
cv2.imwrite(fname + "_sobel_abs.jpg", dst_abs)

dst = cv2.filter2D(gray, cv2.CV_64F, laplacian)
# 結果を出力
cv2.imwrite(fname + "_laplacian.jpg", dst)
# 結果を出力 負の勾配も含む
dst_abs = np.abs(dst)
cv2.imwrite(fname + "_laplacian_abs.jpg", dst_abs)
""""""