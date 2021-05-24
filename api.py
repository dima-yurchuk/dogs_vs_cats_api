import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import prediction

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = "static/"

@app.route('/', methods=['GET','POST'])
def main():
    name = ''
    if request.method=='POST':
        dir = app.config["UPLOAD_FOLDER"] + "images"
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        image = request.files['image']
        filename, file_extension = os.path.splitext(secure_filename(image.filename))
        if image and (file_extension=='.png' or file_extension=='.jpg' or file_extension=='.jpeg'):
            img_name = secure_filename(image.filename)
            image.save(os.path.join(dir, img_name))
            image_class = prediction()
            return render_template('load_image.html', filename=img_name, image_class=image_class)
    return render_template('load_image.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='images/'+ filename))

if __name__ == '__main__':
    app.run(debug=True)
