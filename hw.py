from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/casa")
def hello2():
    return "Hello World! 222"

@app.route("/test")
def circle():
    return render_template('circle.html', name='marco')

if __name__ == "__main__":
    app.run(debug=True)