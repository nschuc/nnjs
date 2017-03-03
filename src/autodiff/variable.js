// @flow
import Tensor from "../tensor.js";
import Op, * as ops from "./ops.js";
import { differentiate } from "./engine.js";

export default class Variable {
  data: Tensor;
  grad: Tensor;
  creator: ?Op;

  constructor(data: Tensor, creator: ?Op) {
    this.data = data;
    this.creator = creator;
  }

  shape(): Array<number> {
    return this.data.shape;
  }

  backward(grad: Tensor = Tensor.ones(1)) {
    differentiate([this], grad);
  }

  *[Symbol.iterator]() {
    for (let i = 0; i < this.data.size[0]; ++i) {
      let op = new ops.Index(i);
      op.setInputs(this);
      yield new Variable(op.forward(this.data), op)
    }
  }

  static _registerOp(name: string, Op) {
    let proto: Object = Variable.prototype;

    proto[name] = function(other: Variable | number) {
      let op = null;
      let data = null;

      if (other instanceof Variable) {
        op = new Op();
        data = other.data;
        op.setInputs(this, other);
      } else if (!other || typeof other == "number") {
        op = new ops.Constant(new Op(), other);
        op.setInputs(this);
      }

      if (op) {
        const result = op.forward(this.data, data);
        return new Variable(result, op);
      } else {
        throw `Error constructing op: ${name}`;
      }
    };
  }
}

Variable.prototype.toString = function() {
  return `Variable with 
    ${this.data.toString()}`;
};

Variable.prototype.inspect = Variable.prototype.toString;
