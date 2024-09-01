import cv2

class isFace:
    def __init__(self,image_path):
        self.cascade_path='haarcascade_frontalface_default.xml'
        self.image_path = image_path

    def count_faces(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self.cascade_path)
        image = cv2.imread(self.image_path)

        if image is None:
            raise Exception("Could not read the image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=18, minSize=(30, 30))

        return len(faces)
    
    def process(self):
        faces_count = self.count_faces()
        if faces_count == 1:
            out = "Face_Detected!"

        elif faces_count > 1:
            out = "Multiple_Faces_Detected!"

        else:
            out = "No_Face_Detected!"

        return out

    

