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

  sub(t2 : Tensor) : Tensor {
    return this._graph.use('sub')({ inputs: [this, t2] });
  }

  pow(exp : number) : Tensor {
    return this._graph.use('pow')({ inputs: [this], exp });
  }

  reduce_sum(attrs : any) : Tensor {
    return this._graph.use('reduce_sum')({ inputs: [this], ...attrs });
  }
}
