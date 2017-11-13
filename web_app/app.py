from flask import Flask, request
app = Flask(__name__)

# Home page
@app.route('/')
def cover():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
