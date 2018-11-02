import pickle
import os
import configparser
from dataset_ocr import dataset_func_OCR
from flask import jsonify
from flask import Flask
from flask import request
from config import config



app = Flask(__name__)

config = configparser.ConfigParser()
config.readfp(open('properties.cfg'))
UPLOAD_FOLDER = config.get('dev','upload_folder')
log_file = config.get('dev','log_file')
log = open(log_file,'w')


my_tfidf = pickle.load(open(config.get('dev','tfidf_pickle_file'), 'rb'))
my_SGD = pickle.load(open(config.get('dev','model_pickle_file'), 'rb'))

@app.route('/func_predict', methods = ["POST"])
def func_predict():
    file = request.files['file'] #contains the filestorage object
    print(file)
    filename = file.filename #filename
    extension = os.path.splitext(filename)[1][1:].lower() #file_extension
    path = os.path.join(UPLOAD_FOLDER, filename) #store the temporary path of the file
    #if extension in ALLOWED_EXTENSIONS: #if the file is among the allowed extensions then proceed
    file.save(path)
    print(path)
    print(filename)
    content = dataset_func_OCR(UPLOAD_FOLDER,filename,log)
    transformed_content = my_tfidf.transform([content])
    y_pred = my_SGD.predict(transformed_content)
    y_prob = my_SGD.predict_proba(transformed_content)
    print(type(y_prob))
    max_prob = max(y_prob[0])*100
    print(y_pred)
    print(max_prob)
    output = {'filename':filename,'prediction':''.join(y_pred),'confidence':max_prob}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
