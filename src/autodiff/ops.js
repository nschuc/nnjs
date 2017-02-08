// @flow
import Tensor from '../tensor.js';

export class Op {
  backwardCache: Array<Tensor>;
  constructor() {
    this.backwardCache = [];
  }

  cacheForBackward(...tensors : Array<Tensor>) {
    this.backwardCache = tensors;
  }

  forward(...inputs : Array<Tensor>) {
    throw "Forward not implemented!"
  }

  backward(grad : Tensor) {
    throw "Backward not implemented!"
  }
}

export class MatMul extends Op {
  forward(t1 : Tensor, t2 : Tensor){
    this.cacheForBackward(t1, t2);
    return t1.mm(t2);
  }

  gradient (grad) {
    let [ t1, t2 ] = this.backwardCache;
    return [ grad.mm(t2), grad.mm(t1) ]
  }
}

export class Add extends Op {
  forward(t1 : Tensor, t2 : Tensor){
    return t1.add(t2);
  }

  gradient (grad : Tensor) {
    return [grad, grad]
  }
}
