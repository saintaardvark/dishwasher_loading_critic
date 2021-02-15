from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/status')
def hello():
    msg = {'status': 'OK', 'msg': 'Hello, world!'}
    return jsonify(msg)
