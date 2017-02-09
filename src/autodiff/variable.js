// @flow
import Tensor from '../tensor.js';
import  Op, * as ops from './ops.js';

export default class Variable {
  data : Tensor;
  grad : Tensor;
  creator : Op;

  constructor(data : Tensor) {
    this.data = data;
  }

  backward(grad : Tensor) {
  
  }

  static _registerOp(name : string, Op) {
    let proto : Object = Variable.prototype;
    proto[name] = function(other : Tensor | Variable) {
      let op = null;

      if(other instanceof Variable) {
        op = new Op();
      }
      else if(!other || typeof other == 'number') {
        op = new ops.Constant(new Op(), other);
      }

      if(op) {
        const t = op.forward(this.data, other.data);
        return new Variable(t);
      }
      else {
        throw `Error constructing op: ${name}`;
      }
    }
  }
}
