from base64 import standard_b64encode

from flask import Flask, jsonify, redirect, render_template, request

from predict import get_prediction
from transform import thumbnailify_image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Main entry point for users
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        try:
            img_bytes = file.read()
        except Exception as e:
            return jsonify({'problem': 'Could not read file', 'error': str(e)})

        class_id, class_name = get_prediction(image_bytes=img_bytes)
        # TODO: Refactor this & thumbnailify_image()
        thumbnail_bytes = str(standard_b64encode(thumbnailify_image(img_bytes)))
        return render_template('result.html', class_id=class_id, class_name=class_name, img_bytes=thumbnail_bytes)

    return render_template('index.html')

@app.route('/status')
def hello():
    msg = {'status': 'OK', 'msg': 'Hello, world!'}
    return jsonify(msg)

# Test: curl -F file=@../dishwasher_training_data/raw/Dishwasher\ Training/20-09-09\ 18-34-11\ 9761.jpg -XPOST http://localhost:5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        try:
            img_bytes = file.read()
        except Exception as e:
            return jsonify({'problem': 'Could not read file', 'error': str(e)})

        class_id, class_name = get_prediction(image_bytes=img_bytes)
        msg = {'class_id': class_id, 'class_name': class_name}
        return jsonify(msg)

if __name__ == '__main__':
    app.run()
