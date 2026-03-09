import cv2
import numpy as np

def extract_features(img):

    # ---- Color Feature ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,12,3],
                        [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    # ---- Texture Feature (GLCM simplified) ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = cv2.calcHist([gray],[0],None,[32],[0,256])
    glcm = cv2.normalize(glcm, glcm).flatten()

    feature = np.concatenate([hist, glcm])

    return feature