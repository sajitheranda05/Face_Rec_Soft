import cv2
import face_recognition
import numpy as np
import base64
import os
import logging
import datetime

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

class RecognizeFace:
    def __init__(self,face_image_path,selfie_image_path):
        self.document_face = cv2.imread(face_image_path)
        self.selfie_face_path = selfie_image_path
        self.UPLOAD_FOLDER = 'temp'


    def extract_face_encodings(self):
        self.selfie_face22 = cv2.imread(self.selfie_face_path)
        rgb_image = cv2.cvtColor(self.document_face, cv2.COLOR_BGR2RGB)
        rgb_image22 = cv2.cvtColor(self.selfie_face22, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image)
        # print(face_locations)
        face_locations22 = face_recognition.face_locations(rgb_image22)
    
        if len(face_locations) == 0:
            logger.error('Response: %s', f"No face detected in the doc_image")
            logger.error('Timestamp: %s', datetime.datetime.now())
            return None
        
        if len(face_locations22) == 0:
            logger.error('Response: %s', f"No face detected in the selfie_image")
            logger.error('Timestamp: %s', datetime.datetime.now())
            return None
        
        if len(face_locations22) > 1:
            face_locations22 = [face_locations22[0]]

        if len(face_locations) > 1:
            face_locations = [face_locations[0]]
        

        
        for (top, right, bottom, left) in face_locations:
            roi2 = self.document_face[top:bottom, left:right]
            retval, buffer = cv2.imencode('.jpg', roi2)
            self.encoded_string = base64.b64encode(buffer).decode('utf-8')

        for (top, right, bottom, left) in face_locations22:
            roi22 = self.selfie_face22[top:bottom, left:right]
            retval22, buffer22 = cv2.imencode('.jpg', roi22)
            self.encoded_string22 = base64.b64encode(buffer22).decode("utf-8")

        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        return face_encodings
    

    def compare_faces(self):

        face_encodings = self.extract_face_encodings()
        bias = 0.4

        if face_encodings is not None:
            selfie_image = face_recognition.load_image_file(self.selfie_face_path)
            selfie_image_encodings = face_recognition.face_encodings(selfie_image)

            if len(selfie_image_encodings) > 0:
                face_distances = face_recognition.face_distance(face_encodings, selfie_image_encodings[0])
                match_results = face_recognition.compare_faces(face_encodings, selfie_image_encodings[0])
        
                best_match_index = face_distances.argmin()
                best_match_result = match_results[best_match_index]
        
                # matching_threshold = 0.5

                Recognition_Data_dict = {}
                if True in match_results:
                    match_probability = 1 - face_distances[best_match_index]
                    # if match_probability >= matching_threshold:
                    result_out = "Successfull"
                    if match_probability < 0.5:
                        probability_out = match_probability + bias 
                    else:
                        probability_out = match_probability 

                    # else:
                    #     result_out = "Face matches, but probability is below the threshold."
                    #     probability_out = match_probability

                else:
                    match_probability = 1 - face_distances[best_match_index]
                    result_out = "Failed"
                    probability_out = match_probability
            else:
                result_out = "No faces found in the selfie image"
                probability_out = 0
        else:
            result_out = "No faces found in the document image"
            probability_out = 0

        logger.info('Response: %s', f"Face detection completed: Result: {result_out}")
        logger.info('Timestamp: %s', datetime.datetime.now())

        return (result_out, probability_out)
    
    def process_file(self):
        res,prob = self.compare_faces()
        return {
            'Result': res,
            'Probability': prob,
        }

    

