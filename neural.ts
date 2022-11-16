import { Value, Valuable } from "./value.ts";

export class Neuron {
  weights: Value[];
  bias: Value;
  /**
   * @param nIn number of inputs of the neuron
   */
  constructor(nIn: number) {
    this.weights = new Array(nIn)
      .fill(0)
      .map(() => new Value(Math.random() * 2 - 1));
    this.bias = new Value(Math.random() * 2 - 1);
  }
  forward(x: Valuable[]) {
    return Value.sum(
      this.weights.map((w, i) => w.multiply(x[i])),
      this.bias
    ).tanh();
  }

  getParameters() {
    return [...this.weights, this.bias];
  }

  zeroGrad() {
    this.getParameters().forEach((p) => (p.grad = 0));
  }

  toString() {
    return `Neuron: with ${this.weights.length} inputs and bias=${this.bias.data}`;
  }
}

export class Layer {
  neurons: Neuron[];
  /**
   * @param nIn number of inputs of each neuron in layer
   * @param nOut number of neurons in layer (one neuron has on output)
   */
  constructor(nIn: number, nOut: number) {
    this.neurons = new Array(nOut).fill(0).map(() => new Neuron(nIn));
  }

  forward(x: Valuable[]) {
    return this.neurons.map((n) => n.forward(x));
  }

  getParameters() {
    return this.neurons.flatMap((n) => n.getParameters());
  }

  zeroGrad() {
    this.neurons.forEach((n) => n.zeroGrad());
  }

  toString() {
    return `Layer: with ${this.neurons.length} neurons of ${this.neurons[0].weights.length} inputs`;
  }
}

export class MultiLayerPerceptron {
  layers: Layer[];

  /**
   * @param nIn number of inputs in first layer
   * @param nOuts array of number of neurons in each layer
   */
  constructor(nIn: number, nOuts: number[]) {
    // inputs of first layer are inputs of the network
    // inputs of each other layers are outputs of previous layer
    const nIns = [nIn, ...nOuts];
    this.layers = nOuts.map((nOut, i) => new Layer(nIns[i], nOut));
  }

  forward(x: Valuable[]) {
    // Layer.forward always returns Value[], thus it the actual produced type of this reduce
    return this.layers.reduce((a, l) => l.forward(a), x) as Value[];
  }

  getParameters() {
    return this.layers.flatMap((l) => l.getParameters());
  }

  zeroGrad() {
    this.layers.forEach((l) => l.zeroGrad());
  }

  toString() {
    return (
      "MultiLayerPerceptron: \n" +
      this.layers.map((l) => l.toString()).join("\n")
    );
  }
}
