// @flow
import Tensor from '../tensor.js';
import * as ops from './ops.js';

export default class Variable {
  data : Tensor;
  grad : Tensor;

  constructor(data : Tensor) {
    this.data = data;
  }

  static _registerOp(name : string, Op) {
    let proto : Object = Variable.prototype;
    proto[name] = function(other : Tensor | Variable) {
      if(other instanceof Variable) {
        const op = new Op();
        const t = op.forward(this.data, other.data);
        return new Variable(t);
      }
    }
  }
}
