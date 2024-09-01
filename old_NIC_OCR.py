from collections import namedtuple
import pytesseract
import cv2
import re
import numpy as np
import json
import logging
import datetime

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

class OCROldNICScan:
  def __init__(self, image):
    self.img = image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    self.average_brightness = cv2.mean(v)[0]
    self.brightness = round(self.average_brightness)
    if self.brightness < 160:
      self.thresh_val = 80

    else:
      self.thresh_val = self.brightness - 20


  def preprocess(self):
    if self.average_brightness > 175:
      thre = 131

    elif self.average_brightness > 160:
      thre = 119

    else:
      thre = 80

    self.gray = cv2.cvtColor(self.img , cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh, self.im_bw = cv2.threshold(self.gray, thre, 255, cv2.THRESH_BINARY)

  def ocrExtract(self):
    custom_config = r'--oem 3 --psm 6 '
    self.ocr = pytesseract.image_to_string(self.img, config= custom_config)

  def extractDate(self):
    self.img = cv2.resize(self.img, (2034,3000))
    OCRLocation = namedtuple("OCRLocation", ["bbox","filter_keywords"])
    OCR_LOCATIONS = [OCRLocation((91, 2224, 652, 356),["BILL", "ACCOUNT", "NUMBER."]),]
    parsingResults = []

    for loc in OCR_LOCATIONS:
        x, y, w, h = loc.bbox
        roi = self.img[y:y+h, x:x+w]


        if roi.size == 0:
            continue  # Skip empty ROI

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = np.ones((1,1), np.uint8)
        eroded_img = cv2.erode(thresh, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        cleaned = cv2.morphologyEx(eroded_img, cv2.MORPH_CLOSE, kernel)
        text = pytesseract.image_to_string(thresh)


        parsingResults.append(text)

    date_string = [s.replace('\\', '1').replace(',', '').replace('°', '.').replace("'", '.').replace(' ', '') for s in parsingResults]
    date_pattern = r"\d{4}.\d{2}.\d{2}"  # Define the pattern as dd/mm/yyyy

    # Check if any string in date_string matches the pattern
    if not any(re.match(date_pattern, s) for s in date_string):
      for loc in OCR_LOCATIONS:
        x, y, w, h = loc.bbox
        roi = self.img[y:y+h, x:x+w]


        if roi.size == 0:
            continue  # Skip empty ROI

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        parsingResults.append(text)
        date_string = [s.replace('\\', '').replace(',', '').replace('°', '.') for s in parsingResults]

    if any(re.match(date_pattern, s) for s in date_string):
      date_string = [s.replace('.', '/') for s in date_string]


    return date_string

  def forceExtractNIC(self):
    height, width = self.img.shape[:2]
    crop_height = int(height / 2)
    cropped_image = self.img[0:crop_height, 0:width]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray[ gray > self.thresh_val] = 255
    gray[ gray < self.thresh_val - 1] = 0
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789Vv'
    text = pytesseract.image_to_string(gray, config = custom_config)
    match = re.search(r'\d{9}\s*[vV]', text)
    extracted_variable = match.group() if match else None

    if not match:
      grayyy = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
      text = pytesseract.image_to_string(grayyy, config = r'--oem 3 --psm 6')
      match = re.search(r'\d{9}\s*[vV]', text)
      extracted_variable = match.group() if match else None

    if not match:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
      closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
      eroded = cv2.erode(gray, kernel, iterations=3)
      text = pytesseract.image_to_string(closed, config = custom_config)
      match = re.search(r'\d{9}\s*[vV]', text)
      extracted_variable = match.group() if match else None

    return (extracted_variable)


  def textProcessing(self):
    ocr_edited = self.ocr.split("\n")

    labeling_patterns = [
      r'\d{9}\s*[vV]',  # Matches 9 digits followed by V
      r'\d{4}\/\d{2}\/\d{2}',  # Matches date in format DD.MM.YYYY
      r'\d{4}\.\d{2}\.\d{2}',
      r'\d{4}\-\d{2}\-\d{2}',
      r'\d{12}',
    ]

    results = []
    for item in ocr_edited:
      for pattern in labeling_patterns:
        match = re.search(pattern, item)
        if match:
          results.append(match.group(0))

    formatted_result = [item.upper() for item in results]
    formatted_result = [item.replace(" ", "") for item in formatted_result]
    formatted_result

    labeled_data = []
    for item in formatted_result:
      if re.search(labeling_patterns[0], item):
        n_item = "NIC: " + item
        labeled_data.append(n_item)

      if re.search(labeling_patterns[4], item):
        n_item = "NIC: " + item
        labeled_data.append(n_item)  

      if re.search(labeling_patterns[1], item):
        n_item = "Issued Date: " + item
        labeled_data.append(n_item)

      if re.search(labeling_patterns[2], item):
        n_item = "Issued Date: " + item
        labeled_data.append(n_item)

    found_issued_date = False

    for element in labeled_data:
      if 'Issued Date:' in element:
        found_issued_date = True
        break

    if not found_issued_date:
      try:
          date_item = self.extractDate()

          results_date = []
          for item in date_item:
              for pattern in labeling_patterns:
                  match = re.search(pattern, item)
                  if match:
                      results_date.append(match.group(0))

          labeled_date = []
          for item in results_date:
              if re.search(labeling_patterns[1], item):
                  n_item = "Issued Date: " + item
                  labeled_date.append(n_item)

              if re.search(labeling_patterns[2], item):
                  n_item = "Issued Date: " + item
                  labeled_date.append(n_item)

          labeled_data = labeled_data + labeled_date

      except AssertionError:
          pass


    if "NIC" not in labeled_data:
      mod_labeled_data = self.forceExtractNIC()
      if mod_labeled_data is not None:
        nic = "NIC: " + mod_labeled_data
        labeled_data.append(nic)


    OldNIC_Data_dict = {}

    for item in labeled_data:
        key_value_pair = item.split(": ")
        OldNIC_Data_dict[key_value_pair[0]] = key_value_pair[1]

    if "NIC" in OldNIC_Data_dict:
         nic = OldNIC_Data_dict["NIC"]
         OldNIC_Data_dict["NIC"] = nic.replace("v", "V").replace(" ", "")
         

    old_NIC = json.dumps(OldNIC_Data_dict)
    return (old_NIC)

  def getOldNIC_OCR(self):
    self.preprocess()
    self.ocrExtract()
    output = self.textProcessing()
    return(output)

  def process_file(self):
    #img = cv2.imread(file_path)  # Read the uploaded image using OpenCV
    result = self.getOldNIC_OCR()
    return result
