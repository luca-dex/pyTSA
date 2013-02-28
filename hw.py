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
def circle():
    numeretti = []
    for _ in range(50):
        numeretti.append([random.randint(1, 400), random.randint(1, 1000)])
    num_to_js = json.dumps(numeretti)
    return render_template('data.html', numb=num_to_js)

if __name__ == "__main__":
    app.run(debug=True)