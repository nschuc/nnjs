// @flow
import Tensor from '../tensor.js';
import  Op, * as ops from './ops.js';
import Engine from './engine.js';

export default class Variable {
  data : Tensor;
  grad : Tensor;
  creator : ?Op;

  constructor(data : Tensor, creator : ?Op) {
    this.data = data;
    this.creator = creator;
  }

  shape() : Array<number> {
    return this.data.shape;
  }

  backward(grad : Tensor = Tensor.ones(1)) {
    this.grad = grad;
    //this._engine.backwards([this], grad);
    if(this.creator) this.creator.run_backward(grad);
  }

  static _registerOp(name : string, Op) {
    let proto : Object = Variable.prototype;

    proto[name] = function(other : Variable | number) {
      let op = null;
      let data = null;

      if(other instanceof Variable) {
        op = new Op();
        data = other.data;
        op.setInputs(this, other);
      }
      else if(!other || typeof other == 'number') {
        op = new ops.Constant(new Op(), other);
        op.setInputs(this);
      }

      if(op) {
        const result = op.forward(this.data, data);
        return new Variable(result, op);
      }
      else {
        throw `Error constructing op: ${name}`;
      }
    }

  }
}
