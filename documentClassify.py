import pytesseract
import cv2

class NICClassify:
    def __init__(self, image):
        self.img = image
        self.output1 = "New_NIC"
        self.output2 = "Old_NIC"

    def checkQuality(self):
        blur_threshold = 10
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        if variance < blur_threshold:
            return True  
        else:
            return False  

    def grayScale(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def OCR(self):
        self.text = pytesseract.image_to_string(self.gray)
        # self.text2 = pytesseract.image_to_string(self.gray, config= r'--oem 3 --psm 6')

    def findClass(self):
        if "NATIONAL" in self.text or "IDENTITY" in self.text or "CARD" in self.text or "SRI LANKA" in self.text or "LANKA" in self.text or "Name" in self.text or "Date" in self.text or "Birth" in self.text or "Holder's" in self.text or "Signature" in self.text:
            self.classified = True
            return self.output1

        else:
            self.classified = False
            return self.output2

    def classify(self):
        self.grayScale()
        self.OCR()
        output = self.findClass()
        return output

    def process_file(self, file_path):
        self.img = cv2.imread(file_path)  
        pre_result = self.classify()
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        self.average_brightness = cv2.mean(v)[0]

        if self.average_brightness < 25:
            self.classified = pre_result
            result = self.output8

        else:
            if pre_result == "NIC" in pre_result:
                result = self.classify()
            
            else:
                is_blurry_image = self.checkQuality()
                if is_blurry_image:
                    self.classified = pre_result
                    result = self.output7
                else:
                    result = self.classify()

        return result 