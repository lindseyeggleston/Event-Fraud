from flask import Flask, request, render_template
app = Flask(__name__)

# Home page
@app.route('/')
def cover():
    return render_template('cover_index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
