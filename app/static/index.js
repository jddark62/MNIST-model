let s = null;
let pre = null;

function setup() {
  let canvas = createCanvas(200, 200);
  canvas.parent("Canvas");
  background(255);
  let predict = select("#predict");
  predict.mousePressed(guessTheNumber);
  let cb = select("#clearB");
  cb.mousePressed(cc);
  pre = select("#pred");
}

function guessTheNumber() {
  let inputs = [];
  let img = get();
  img.resize(28, 28);
  img.loadPixels();
  for (let i = 0; i < 28 * 28; i++) {
    let bright = img.pixels[i * 4];
    inputs[i] = (255 - bright) / 255.0;
  }
  httpPost("/predict", "json", { input: inputs }, (result) => {
    pre.html(result["prediction"]);
  });
}

function cc() {
  clear();
  background(255);
}

function draw() {
  stroke(0);
  strokeWeight(8);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}
