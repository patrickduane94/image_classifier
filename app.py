from flask import Flask, render_template, request
import os
from model import load_model, predict_image
import base64


app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    image_data = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            prediction, probability = predict_image(filepath, model)

            with open(filepath, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template('index.html', prediction=prediction, image_data=image_data, probability=probability)


if __name__ == '__main__':
    app.run(debug=True)
