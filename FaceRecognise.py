# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:22:51 2018

@author: Greenu
"""

import face_recognition # for facial recognition
import cv2  # for Image read and processing
import winsound # For alarm sound
import pyttsx3 # for Text to speech
import os
import sys


def main(inputfolderpath):
    
    # This is a demo of running face recognition on live video from your webcam. 
    #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
    #   2. Only detect faces in every other frame of video.
      
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    
    #Enable this line for a network IP camera.Update with the specific IP
    #video_capture = cv2.VideoCapture('rtsp://192.168.1.88/1')
       
    # Load a sample picture and learn how to recognize it.
    # Get the reference images from the Input Directory
    # File Names(exclusing extension) will be used as the label for each images
    inputfolder=inputfolderpath
    #print(inputfolderpath)
    #print(inputfolder)
    if not inputfolder:
       #print('False')
       #inputfolder=os.path.dirname(os.path.abspath(__file__)) + '\*.jpg'
       sys.exit("Please provide valid input")
    #ref_images = [cv2.imread(file) for file in glob.glob(inputfolder)]
    if isinstance(inputfolder, list)==False:
        sys.exit("Please provide valid input")
    known_face_names=[]
    known_face_encodings=[]
    print('Reading files from the input directory ' )
    for img in inputfolder:
        print('Image Encoding')
        face_image=face_recognition.load_image_file(img)
        face_encoding = face_recognition.face_encodings(face_image)
        print('Image appending')
        known_face_encodings.extend(face_encoding)
        root, ext =os.path.splitext(os.path.basename(img))
        known_face_names.append(root)
    
    
     
    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    engine=pyttsx3.init()
    #Welcome message for your program , in case needed
    engine.say("Hello, Welcome to Face recognition program")
    engine.runAndWait()
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
    
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
    
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # If a match was found in known_face_encodings, just use the first one.
                if (True in matches):
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
    
                face_names.append(name)
    
        process_this_frame = not process_this_frame
    
    
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
    
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            if (name == "Unknown"):
            # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                winsound.Beep(frequency, duration)
            else:
                 cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
        # Display the resulting image
        cv2.imshow('Video', frame)
       # engine.say("Hello" + name)
       # engine.runAndWait()
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release handle to the webcam
    video_capture.release()
    print('Video released')
    print('Program Exiting.....')
    cv2.destroyAllWindows()
    print('Terminated!')

if __name__ == "__main__":
   main(sys.argv[1:])