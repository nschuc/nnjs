// @flow
import Tensor from "../tensor.js";
import Variable from "./variable.js";

export default class Op {
  backwardCache: Array<Tensor>;
  inputs: Array<Variable>;

  constructor() {
    this.backwardCache = [];
    this.inputs = [];
  }

  cacheForBackward(...tensors: Array<any>) {
    this.backwardCache = tensors;
  }

  forward(t1: Tensor, t2: Tensor | number) {
    throw "Forward not implemented!";
  }

  backward(grad: Tensor): Array<any> {
    throw "Backward not implemented!";
  }

  setInputs(...inputs: Array<Variable>) {
    this.inputs = inputs;
  }

  run_backward(grad: Tensor) {
    const grads = this.backward(grad);
    for (let i = 0; i < this.inputs.length; i++) {
      this.inputs[i].backward(grads[i]);
    }
  }
}

export class MatMul extends Op {
  forward(t1: Tensor, t2: Tensor | number) {
    if (typeof t2 == "number") {
      throw "Error: cannot perform matrix multiplication with a number";
    }
    this.cacheForBackward(t1, t2);
    return t1.mm(t2);
  }

  backward(grad: Tensor) {
    let [t1, t2] = this.backwardCache;
    return [grad.mm(t2.T), t1.T.mm(grad)];
  }
}

export class Add extends Op {
  forward(t1: Tensor, t2: Tensor | number) {
    return t1.add(t2);
  }

  backward(grad: Tensor): Array<Tensor> {
    return [grad, grad];
  }
}

export class Sub extends Op {
  forward(t1: Tensor, t2: Tensor | number) {
    return t1.sub(t2);
  }

  backward(grad: Tensor): Array<Tensor> {
    return [grad, grad.neg()];
  }
}

export class Mul extends Op {
  forward(t1: Tensor, t2: Tensor | number) {
    this.cacheForBackward(t1, t2);
    return t1.mul(t2);
  }

  backward(grad: Tensor) {
    let [t1, t2] = this.backwardCache;
    return [grad.mul(t2), grad.mul(t1)];
  }
}

export class Pow extends Op {
  forward(t1: Tensor, p: number) {
    this.cacheForBackward(t1);
    this.p = p;
    return t1.pow(p);
  }

  backward(grad: Tensor) {
    let [t1] = this.backwardCache;
    return [grad.mul(this.p).mul(t1.pow(this.p - 1))];
  }
}

export class Norm extends Op {
  forward(t1: Tensor) {
    this.cacheForBackward(t1);
    return t1.norm();
  }

  backward(grad: Tensor) {
    let [t1] = this.backwardCache;
    return [t1];
  }
}

export class Neg extends Op {
  forward(t1: Tensor) {
    return t1.neg();
  }

  backward(grad: Tensor) {
    return [grad.neg()];
  }
}

export class Constant extends Op {
  value: number;
  op: Op;
  constructor(op: Op, value: number) {
    super();
    this.value = value;
    this.op = op;
  }

  forward(t1: Tensor) {
    return this.op.forward(t1, this.value);
  }

  backward(grad: Tensor) {
    const grads = this.op.backward(grad);
    return [grads[0]];
  }
}

export class Index extends Op {
  index: number;
  constructor(index: number) {
    super();
    this.index = index;
  }

  forward(t1: Tensor) {
    this.size = t1.size;
    return t1.index(this.index);
  }

  backward(grad: Tensor) {
    let input_grad = Tensor.zeros(...this.size);
    input_grad.set_index_(this.index, grad);
    return [input_grad];
  }
}

export class Dist extends Op {
  constructor() {
    super();
  }

  forward(t1: Tensor, t2: Tensor) {
    return t1.dist(t2);
  }

  backward(grad: Tensor) {
    throw "Backward not implemented!"
  }
}
