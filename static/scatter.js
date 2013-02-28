//Width and height
			var w = 1000;
			var h = 400;
			
			//var dataset = [
			//				[5, 20], [480, 90], [250, 50], [100, 33], [330, 95],
			//				[410, 12], [475, 44], [25, 67], [85, 21], [220, 88]
			//			  ];
	
			//Create SVG element
function scatter(dataset){
			var svg = d3.select("#test")
						.append("svg")
						.attr("width", w)
						.attr("height", h);

			svg.selectAll("circle")
			   .data(dataset)
			   .enter()
			   .append("circle")
			   .attr("cx", function(d) {
			   		return d[0];
			   })
			   .attr("cy", function(d) {
			   		return d[1];
			   })
			   .attr("r", function(d) {
			   		return 3;
			   });

			svg.selectAll("text")
			   .data(dataset)
			   .enter()
			   .append("text")
			   .text(function(d) {
			   		return d[0] + "," + d[1];
			   })
			   .attr("x", function(d) {
			   		return d[0];
			   })
			   .attr("y", function(d) {
			   		return d[1];
			   })
			   .attr("font-family", "sans-serif")
			   .attr("font-size", "11px")
			   .attr("fill", "red");   
}