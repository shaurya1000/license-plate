import cv2 # Importing necessary library
import numpy as np # Importing necessary library
import pytesseract # Importing necessary library
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract' # Configuring Tesseract to use the number plate configuration
haarcascade_path = 'number_plate.xml'
plat_detector = cv2.CascadeClassifier(haarcascade_path) # Loading the Haar cascade file for number plate detection
video_path = 'S-vid.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print('Error Reading Video')
else:
    while True:
        ret, frame = video.read()
        if ret:
            gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            plates = plat_detector.detectMultiScale(gray_video, scaleFactor=1.2, minNeighbors=5, minSize=(25,25))

            for (x, y, w, h) in plates: #Using for loop
                roi_gray = gray_video[y:y+h, x:x+w]   # Extracting the region of interest 
                text = pytesseract.image_to_string(roi_gray, config='--psm 8')
                print('Detected Number Plate:', text.strip()) 

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)    # Drawing a rectangle around the plate and optionally blur it
                frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], ksize=(10, 10))
                cv2.putText(frame, 'License Plate', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
            # Displaying the resulting frame in a window called 'Video'
            cv2.imshow('Video', frame)
            # Breaking the loop when 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()