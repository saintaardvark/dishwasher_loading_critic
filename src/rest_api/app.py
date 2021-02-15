from flask import Flask, jsonify, redirect, request

from predict import get_prediction

app = Flask(__name__)

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
