import { MultiLayerPerceptron } from "./neural.ts";
import { Value } from "./value.ts";

// example from https://www.youtube.com/watch?v=VMj-3S1tku0&t=6664s
// training data
const xs = [
  [2, 3, -1],
  [3, -1, 0.5],
  [0.5, 1, 1],
  [1, 1, -1],
];

// labels: expected outputs for each row in training data
const ys = [1, -1, -1, 1];

const network = new MultiLayerPerceptron(3, [
  [4, "tanh"],
  [4, "tanh"],
  [1, "tanh"],
]);

const learningRate = 0.07;
function evaluateLoss() {
  const yPred = xs.flatMap((x) => network.forward(x));
  const squareErrors = yPred.map((y, i) => y.subtract(ys[i]).power(2));
  return Value.sum(squareErrors);
}

console.log("Training:");
let loss = evaluateLoss();
const history = [];
while (loss.data > 0.005) {
  loss = evaluateLoss();
  history.push(loss.data);
  loss.backward();

  const params = network.getParameters();

  params.forEach((p) => {
    p.data += -learningRate * p.grad;
  });
  network.zeroGrad();
}
console.log("loss: " + history.join("\nloss: "));
console.log(`Trained for ${history.length} epochs`);

const yPred = xs.flatMap((x) => network.forward(x));
console.log("Resulting predictions: " + yPred.map((y) => y.data).join(","));
