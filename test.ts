import { assertEquals, assertAlmostEquals } from "https://deno.land/std@0.163.0/testing/asserts.ts";

import { Value } from "./value.ts";

Deno.test("Value: initial gradient", () => {
  const value = new Value(1);
  assertEquals(value.grad, 0, `Initial gradient should be 0; ${value.grad} given`);
});


// L = (c + (a * b)) * f – first example from Andrej Karpathy's manual backprop exercise
Deno.test("Value: forward path", () => {
    const a = new Value(2);
    const b = new Value(-3);
    const c = new Value(10);
    const e = a.multiply(b);
    const d = c.add(e);
    const f = new Value(-2);
    const L = d.multiply(f);

    assertEquals(e.data, -6);
    assertEquals(d.data, 4);
    assertEquals(L.data, -8);
})

Deno.test("Value: backpropagation", () => {
    const a = new Value(2);
    const b = new Value(-3);
    const c = new Value(10);
    const e = a.multiply(b);
    const d = c.add(e);
    const f = new Value(-2);
    const L = d.multiply(f);
    L.backward()

    assertEquals(L.grad, 1, `L gradient should be 1, ${L.grad} given`);
    assertEquals(d.grad, -2, `d gradient should be -2, ${d.grad} given`);
    assertEquals(f.grad, 4, `f gradient should be 4, ${f.grad} given`);
    assertEquals(e.grad, -2, `e gradient should be -2, ${e.grad} given`);
    assertEquals(c.grad, -2, `c gradient should be -2, ${c.grad} given`);
    assertEquals(a.grad, 6, `a gradient should be 1, ${a.grad} given`);
    assertEquals(b.grad, -4, `b gradient should be 1, ${b.grad} given`);
});

// L = tanh(x1 * w1 + x2 * w2 + b) – second example of manual backprop exercise
Deno.test("Neuron: backpropagation", () => {
    // inputs x1,x2
    const x1 = new Value(2);
    const x2 = new Value(0);
    // weights w1,w2
    const w1 = new Value(-3);
    const w2 = new Value(1);
    //  bias of the neuron
    const b = new Value(6.8813735870195432);
    // x1*w1 + x2*w2 + b
    const x1w1 = x1.multiply(w1);
    const x2w2 = x2.multiply(w2);
    const sum = x1w1.add(x2w2);
    const n = sum.add(b);

    const o = n.tanh();
    o.backward();

    const t =  0.00001 // tolerance
    assertEquals(o.grad, 1, `o gradient should be 1, ${o.grad} given`);
    assertAlmostEquals(n.grad, 0.5, t, `n gradient should be 0.5, ${n.grad} given`);
    assertAlmostEquals(sum.grad, 0.5, t, `sum gradient should be 0.5, ${sum.grad} given`);
    assertAlmostEquals(x1w1.grad, 0.5, t, `x1w1 gradient should be 0.5, ${x1w1.grad} given`);
    assertAlmostEquals(x2w2.grad, 0.5, t, `x2w2 gradient should be 0.5, ${x2w2.grad} given`);
    assertAlmostEquals(x1.grad, -1.5, t, `x1 gradient should be -1.5, ${x1.grad} given`);
    assertAlmostEquals(x2.grad, 0.5, t, `x2 gradient should be 0.5, ${x2.grad} given`);
    assertAlmostEquals(w1.grad, 1, t, `w1 gradient should be 1, ${w1.grad} given`);
    assertEquals(w2.grad, 0, `w2 gradient should be 0, ${w2.grad} given`);
})