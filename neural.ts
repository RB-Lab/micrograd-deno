import { Value, Valuable } from "./value.ts";

export interface Module {
  forward(x: Valuable[]): Value[] | Value;
  zeroGrad(): void;
  getParameters(): Value[];
}

type Activation = "relu" | "tanh";

export class Neuron implements Module {
  weights: Value[];
  bias: Value;
  activation?: Activation;
  /**
   * @param nIn number of inputs of the neuron
   * @param activation activation function to use
   */
  constructor(nIn: number, activation?: Activation) {
    this.weights = new Array(nIn)
      .fill(0)
      .map(() => new Value(Math.random() * 2 - 1));
    this.bias = new Value(Math.random() * 2 - 1);
    this.activation = activation;
  }
  forward(x: Valuable[]) {
    const sum = Value.sum(
      this.weights.map((w, i) => w.multiply(x[i])),
      this.bias
    );
    return this.activation ? sum[this.activation]() : sum;
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

export class Layer implements Module {
  neurons: Neuron[];
  activation?: Activation;
  /**
   * @param nIn number of inputs of each neuron in layer
   * @param nOut number of neurons in layer (one neuron has on output)
   * @param activation activation function of layer's neurons
   */
  constructor(nIn: number, nOut: number, activation?: Activation) {
    this.neurons = new Array(nOut)
      .fill(0)
      .map(() => new Neuron(nIn, activation));
    this.activation = activation;
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

/**
 * Layer descriptor. The first element in tuple is the number of neurons in the layer,
 * the second element is the activation function.
 */
type LayerDescriptor = [number] | [number, Activation];

export class MultiLayerPerceptron implements Module {
  layers: Layer[];

  /**
   * @param nIn number of inputs in first layer
   * @param layers array of layer descriptors
   */
  constructor(nIn: number, layers: LayerDescriptor[]) {
    // inputs of first layer are inputs of the network
    // inputs of each other layers are outputs of previous layer
    const nIns = [nIn, ...layers.map((l) => l[0])];
    this.layers = layers.map((l, i) => new Layer(nIns[i], l[0], l[1]));
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
