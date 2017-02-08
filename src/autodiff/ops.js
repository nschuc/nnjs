// @flow
import Tensor from '../tensor.js';

export default class Op {
  backwardCache: Array<Tensor>;
  constructor() {
    this.backwardCache = [];
  }

  cacheForBackward(...tensors : Array<any>) {
    this.backwardCache = tensors;
  }

  forward(t1 : Tensor, t2 : Tensor | number) {
    throw "Forward not implemented!"
  }

  backward(grad : Tensor) {
    throw "Backward not implemented!"
  }
}

export class MatMul extends Op {
  forward(t1 : Tensor, t2 : Tensor | number){
    if(typeof t2 == 'number') {
      throw 'Error: cannot perform matrix multiplication with a number';
    }
    this.cacheForBackward(t1, t2);
    return t1.mm(t2);
  }

  gradient (grad : Tensor) {
    let [ t1, t2 ] = this.backwardCache;
    return [ grad.mm(t2.T), grad.mm(t1.T) ]
  }
}

export class Add extends Op {
  forward(t1 : Tensor, t2 : Tensor | number){
    return t1.add(t2);
  }

  gradient (grad : Tensor) {
    return [grad, grad]
  }
}

export class Mul extends Op {

  forward(t1 : Tensor, t2 : Tensor | number){
    this.cacheForBackward(t1, t2);
    return t1.mul(t2);
  }

  gradient (grad : Tensor) {
    let [ t1, t2 ] = this.backwardCache;
    return [ grad.mul(t2), grad.mul(t1) ]
  }
}

export class Norm extends Op {

  forward(t1 : Tensor){
    this.cacheForBackward(t1);
    return t1.norm();
  }

  gradient (grad : Tensor) {
    let [ t1 ] = this.backwardCache;
    return [ t1 ]
  }
}

export class Constant extends Op{
  value : number;
  op : Op;
  constructor(op : Op, value : number) {
    super();
    this.value = value;
    this.op = op;
  }

  forward(t1 : Tensor){
    return this.op.forward(t1, this.value);
  }

  gradient (grad : Tensor) {
    const grads = this.op.backward(grad);
    return grads[0]
  }

}
