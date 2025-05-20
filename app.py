from flask import Flask, render_template, request
import os
from model import load_model, predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            prediction, probability = predict_image(filepath, model)
            image_url = filepath

    return render_template('index.html', prediction=prediction, image_url=image_url, probability=probability)


if __name__ == '__main__':
    app.run(debug=True)
