function circle(){
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
}
