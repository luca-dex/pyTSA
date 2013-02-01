from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/casa")
def hello2():
    return "Hello World! 222"

@app.route("/test")
def circle():
    return '''
	<!DOCTYPE html>
	<html>
 		<head>
		<title>Hello, data!</title>
    <script src="http://d3js.org/d3.v3.min.js"></script>
  </head>
  <body>

    <div id="viz"></div>

    <script type="text/javascript">

    var sampleSVG = d3.select("#viz")
        .append("svg:svg")
        .attr("width", 100)
        .attr("height", 100);

    sampleSVG.append("svg:circle")
        .style("stroke", "black")
        .style("fill", "white")
        .attr("r", 40)
        .attr("cx", 50)
        .attr("cy", 50)

    </script>

  </body>
</html>




    '''

if __name__ == "__main__":
    app.run(debug=True)