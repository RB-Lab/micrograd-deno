export class Value {
  private backwards_ = () => {};
  public grad = 0;
  constructor(
    public data: number,
    private children: [Value, Value] | [Value] | [] = [],
    private operation: string = ""
  ) {}

  add(other_: Value | number) {
    const other = this.wrap(other_);
    const out = new Value(this.data + other.data, [this, other], "+");
    out.backwards_ = () => {
      // this is an application of the chain rule:
      // because a "local derivative" of the addition is 1,
      // we just copy the gradient from the output to the inputs
      // += instead of = here in case this node was called multiple times
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  multiply(other_: Value | number) {
    const other = this.wrap(other_);
    const out = new Value(this.data * other.data, [this, other], "*");
    out.backwards_ = () => {
      // "local derivative" of multiplication is the value of the other node
      // so, applying the chain rule, we multiply the derivative of the outer/next
      // expression by the other node's value
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  exp() {
    const out = new Value(Math.exp(this.data), [this], "exp");
    out.backwards_ = () => {
      // derivative of exp(x) is exp(x)
      this.grad += out.grad * Math.exp(this.data);
    };
    return out;
  }

  power(exponent: number) {
    const out = new Value(Math.pow(this.data, exponent), [this], "^");
    out.backwards_ = () => {
      this.grad += exponent * Math.pow(this.data, exponent - 1) * out.grad;
    };
    return out;
  }

  div(other_: Value | number) {
    const other = this.wrap(other_);
    return this.multiply(other.power(-1));
  }

  tanh() {
    const out = new Value(Math.tanh(this.data), [this], "tanh");
    out.backwards_ = () => {
      this.grad += (1 - Math.pow(Math.tanh(this.data), 2)) * out.grad;
    };
    return out;
  }

  backward() {
    // sort all nodes topologically
    const visited = new Set<Value>();
    const stack: Value[] = [];
    const visit = (node: Value) => {
      if (visited.has(node)) return;
      visited.add(node);
      node.children.forEach(visit);
      stack.push(node);
    }
    visit(this);
    this.grad = 1;
    // now compute gradients in reverse order
    stack.reverse().forEach((node) => node.backwards_());
  }

  toString() {
    if (this.children.length === 0) {
      return `Value: (${this.data})`;
    } else if (this.children.length === 1) {
      return `Value: (${this.data}) ${this.operation} (${this.children[0].data})`;
    } else {
      return `Value: (${this.data}) ${this.children
        .map((c) => c.data)
        .join(` ${this.operation} `)}`;
    }
  }

  /**
   * in case we get a number instead of a Value, wrap it in a Value
   */
  private wrap(other: Value | number) {
    if (other instanceof Value) {
      return other;
    }
    return new Value(other);
  }
}
