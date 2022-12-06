import {
  loadMnist,
  shuffle,
  normalize,
  downscaleImage,
  printDigit,
} from "https://deno.land/x/mnist@v1.1.0/mod.ts";
import { MultiLayerPerceptron } from "./neural.ts";
import { Value } from "./value.ts";

const { train, test } = await loadMnist();

const network = new MultiLayerPerceptron(196, [[32, "relu"], [10]]);

function getMiniBatch<T>(size: number, set: T[]) {
  const shuffled = shuffle(set);
  return shuffled.slice(0, size);
}

function oneHot(digit: number) {
  const vector = new Array(10).fill(0);
  vector[digit] = 1;
  return vector;
}

function argMax(ns: number[]) {
  const max = Math.max(...ns);
  return ns.indexOf(max);
}

function evaluateLoss() {
  const batch = getMiniBatch(200, train);
  const yPred = batch
    .map((d) => normalize(downscaleImage(d.image)))
    .map((x) => network.forward(x))
    .map((y) => Value.softmax(y));
  const squareErrors = yPred.map((yPredVec, i) => {
    const yVec = oneHot(batch[i].label);
    return yPredVec.map((y, j) => y.subtract(yVec[j]).power(2));
  });

  return Value.sum(squareErrors.flatMap((e) => e));
}

console.log("Training:");
console.log("total params", network.getParameters().length);
console.log("epoch\tloss\ttime");
let loss = evaluateLoss();
const threshold = loss.data * 0.1;
let epoch = 0;
let learningRate = 0.03;
const thresholdCrossings: number[] = [];
while (thresholdCrossings.length < 5) {
  if (loss.data < threshold) thresholdCrossings.push(loss.data);
  else thresholdCrossings.length = 0;
  learningRate *= 0.995; // learning rate decay
  performance.mark("epoch-start");
  loss = evaluateLoss();
  loss.backward();
  const params = network.getParameters();

  params.forEach((p) => {
    p.data += -learningRate * p.grad;
  });
  network.zeroGrad();
  performance.mark("epoch-end");
  const takes = performance.measure("epoch", "epoch-start", "epoch-end");
  console.log(
    `${epoch}\t${loss.data.toFixed(2)}\t${(takes.duration / 1000).toFixed(2)}}`
  );
  epoch++;
}

const testBatch = getMiniBatch(100, test);
const yPredBatch = testBatch
  .map((d) => normalize(downscaleImage(d.image)))
  .map((x) => network.forward(x))
  .map((y) => Value.softmax(y));

const accuracy =
  yPredBatch.filter((yPred, i) => {
    const y = testBatch[i].label;
    return argMax(yPred.map((v) => v.data)) === y;
  }).length / yPredBatch.length;

console.log("Accuracy:", accuracy);

testBatch.slice(0, 10).forEach((d, i) => {
  const yPred = yPredBatch[i];
  console.log(
    printDigit(downscaleImage(d.image)),
    `\n ${argMax(yPred.map((v) => v.data))} <- ${d.label}\n`,
    yPred.map((y) => y.data.toFixed(2)).join(", ")
  );
});
