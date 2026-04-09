import cv2

cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
print("Loaded:", not cascade.empty())