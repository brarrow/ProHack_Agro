import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
# cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

def run(img):
    # Read the frame
    # _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.01, 4)
    # Draw the rectangle around each face

    max_index = max(faces, key=lambda x: x[2] * x[3])

    return max_index

    # for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


    # Display
    # cv2.imshow('img', img)
    # cv2.waitKey(1)

# Release the VideoCapture object
# cap.release()