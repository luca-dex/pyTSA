from flask import Flask, render_template, url_for
import json, random

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/test")
def circle():
    return render_template('circle.html', name='mr. x')

@app.route("/data")
def data():
    numeretti = []
    numeretti2 = []
    numeretti3 = []
    for _ in range(5):
        numeretti.append([random.randint(1, 1000), random.randint(1, 600)])
        numeretti2.append([random.randint(1, 1000), random.randint(1, 600)])
        numeretti3.append([random.randint(1, 1000), random.randint(1, 600)])
    num_to_js = json.dumps({'serie1': numeretti, 'serie2': numeretti2, 'serie3': numeretti3}, indent=4, separators=(',',': '))
    return render_template('data.html', numb=num_to_js)

if __name__ == "__main__":
    app.run(debug=True)