//@flow
import Graph from './graph';
import { Op } from './ops';
import ndarray from 'ndarray';

export type Shape = Array<number>

export default class Tensor {
  input: Op;
  _graph: Graph;

  constructor(op: Op, _graph : Graph) {
    this.input = op;
    this._graph = _graph;
  }

  getShape() : Shape {
    return this.input.shape;
  }

  getId() : string {
    return this.input.id;
  }

  mm(t2 : Tensor) : Tensor {
    return this._graph.use('mm')({ inputs: [this, t2] });
  }

  plus(t2 : Tensor) : Tensor {
    return this._graph.use('plus')({ inputs: [this, t2] });
  }
}
