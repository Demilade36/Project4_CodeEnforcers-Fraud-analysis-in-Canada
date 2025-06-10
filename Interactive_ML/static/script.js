document.getElementById("graphSelector").addEventListener("change", function() {
    let selectedGraph = this.value;
    document.getElementById("graphImage").src = "/static/" + selectedGraph + ".png";
});
