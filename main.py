import cv2
import numpy as np

# Constantes
ESC_KEY = 27  # Code ASCII pour la touche ESC
SPC_KEY = 32  # Code ASCII pour la touche ESPACE

def calibration(images, board_size):
    # Préparer les points d'objet comme (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # Tableaux pour stocker les points d'objet et les points d'image
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan de l'image

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Trouver les coins de l'échiquier
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            objpoints.append(objp)
            # Affiner les positions des coins
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_ITER, 30, 0.1)
            )
            imgpoints.append(corners2)

            # Dessiner et afficher les coins
            cv2.drawChessboardCorners(img, board_size, corners2, ret)
            cv2.imshow('Coins de l\'échiquier', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistort_image(image, mtx, dist):
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Déformer
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

    # Recadrer l'image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


if __name__ == "__main__":
    # Initialiser la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra")
        exit()
    
    # Créer une fenêtre
    window_name = "Calibration de la camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Variables pour le basculement en niveaux de gris
    is_grayscale = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de recevoir l'image (fin du flux?). Sortie ...")
            break

        # Convertir en niveaux de gris si le basculement est actif
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = gray_frame if is_grayscale else frame

        # Afficher l'image
        cv2.imshow(window_name, display_frame)

        # Vérifier l'entrée utilisateur
        key = cv2.waitKey(1)
        if key == ESC_KEY:  # Sortir sur ESC
            break
        elif key == ord('g'):  # Basculer en niveaux de gris sur 'g'
            is_grayscale = not is_grayscale

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
