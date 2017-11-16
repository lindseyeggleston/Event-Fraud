from flask import Flask, request, render_template
app = Flask(__name__)

# Home page
@app.route('/')
def cover():
    return render_template('cover_index.html')

# Cost/benefit page
@app.route('/cost_benefit', methods=['GET', 'POST'])
def cost_benefit():
    # benefit = int(request.form['Average savings (per case) of correct early fraud detection:'])
    # cost = int(request.form['Average cost (per case) of investigating possible fraud:'])
    return render_template('cover_cost-benefit.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
