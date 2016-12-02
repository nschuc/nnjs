//@flow
import Graph from './graph';
import { Op } from './ops';
import ndarray from 'ndarray';

export type Shape = Array<number>

export default class Tensor {
  input: Op;
  outputs: Array<Op>;
  graph: Graph;

  constructor(op: Op, graph : Graph) {
    this.input = op;
    this.graph = graph;
    this.outputs = [];
    this.input.outputs.push(this);
  }

  getShape() : Shape {
    return this.input.getShape();
  }

  getId() : string {
    return this.input.id;
  }

  _result() {
    return this.input.result;
  }

  mm(t2 : Tensor) : Tensor {
    return this.graph.use('mm')({ inputs: [this, t2] });
  }

  plus(t2 : Tensor) : Tensor {
    return this.graph.use('plus')({ inputs: [this, t2] });
  }
}
