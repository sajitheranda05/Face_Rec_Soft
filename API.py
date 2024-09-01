from flask import Flask, request, jsonify
import os
import cv2
from documentClassify import NICClassify
from new_NIC_OCR import OCRNewNICScan
from old_NIC_OCR import OCROldNICScan
import json
from Age_Calculator import AgeCalculator
import logging
from getFaceEncodings import RecognizeFace
from face_recog_class import isFace

app = Flask(__name__)

UPLOAD_FOLDER = 'temp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/ageVerification', methods=['POST'])
def upload_files():
    if 'NIC' not in request.files or 'selfieImage' not in request.files:
        return jsonify({'error': 'Files "NIC" and "selfie image" are required.'}), 400
    
    nic_file = request.files['NIC']
    selfie_image_file = request.files['selfieImage']
    logging.info('Documents Received as Input')
    
    nic_filename = os.path.join(app.config['UPLOAD_FOLDER'], nic_file.filename)
    selfie_image_filename = os.path.join(app.config['UPLOAD_FOLDER'], selfie_image_file.filename)
    nic_file.save(nic_filename)
    selfie_image_file.save(selfie_image_filename)
    logging.info("NIC and selfie image files saved successfully.")

    nic_image = cv2.imread(nic_filename)
    selfie_image = cv2.imread(selfie_image_filename)

    classify_nic = NICClassify(nic_image)
    doc_type  = classify_nic.process_file(nic_filename)
    logging.info(f"Classified Document Type: {doc_type}")

    if doc_type == "New_NIC":
        new_ocr = OCRNewNICScan(nic_image)
        text = new_ocr.process_file()
        nic_no = text['NIC']
        logging.info(f"Extracted NIC Number: {nic_no}")

    elif doc_type == "Old_NIC":
        new_ocr = OCROldNICScan(nic_image)
        text = new_ocr.process_file()
        detail_dict = json.loads(text)
        nic_no = detail_dict.get('NIC', None)
        logging.info(f"Extracted NIC Number: {nic_no}")
    else:
        logging.error("Unsupported document type detected.")

    
    calculator = AgeCalculator(nic_no)
    age = calculator.calculate()
    logging.info(f"Calculated Age: {age}")

    classifier = isFace(selfie_image_filename)   
    result = classifier.process()   
    if result == "Face_Detected!":
            resultStatus = "00"
            description = "Successfull" 
            error = None
            logging.info(f"Face Detection: {description}")

    elif result == "Multiple_Faces_Detected!":
            resultStatus = "02"
            description = "Failed"
            error = "Multiple_Faces_Detected"
            logging.error("Multiple Faces Detected")
            
    else:
            resultStatus = "01"
            description = "Failed"
            error = "Face Detection Failed"
            logging.error("Face Detection Failed")

    if resultStatus == "00":
        classifier = RecognizeFace(nic_filename, selfie_image_filename)
        result = classifier.process_file()
        res = result['Result']
        prob = result['Probability']

    elif resultStatus == "02" or resultStatus == "01":
        res = "Failed"
        prob = 0

    logging.info(f"Face Verification Completed: Result -> {res}, Probability -> {prob}")

    if age is None:
         error = "NIC Capturing Failed, Re-upload a Clear Image"


    if age is not None and description == "Successfull" and res == "Successfull":
         age_ver = "Pass"
    else:
        age_ver = "Fail" 

    response = {
                'Age': age,
                'Overall Result': age_ver,
                'Face Detection': description,
                'Face Verification': res,
                'Face Verification Probability': prob,
                'Issues': error
                }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
