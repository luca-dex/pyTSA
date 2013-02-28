//Width and height
			var w = 1500;
			var h = 700;
			
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
	var color = d3.scale.category10()
		
		circle = svg.selectAll("circle");
		for(var serie in dataset) {
			circle.data(dataset[serie])
			.enter()
			.append("circle")
			   .attr("cx", function(d) {
			   		return d[0];
			   })
			   .attr("cy", function(d) {
			   		return d[1];
			   })
			   .attr("r", function(d) {
			   		return 5;
			   })
			   .style("fill", function(d) { 	
		   			return color(serie); 
			   });
	}

		text = svg.selectAll("text");
		for(var serie in dataset) {
			text.data(dataset[serie])
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

}