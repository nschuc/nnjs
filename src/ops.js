// @flow
import Graph from './graph';
import Tensor from './tensor';
import type { Shape } from './tensor';

import ndarray from 'ndarray';
import gemm from 'ndarray-gemm';

export type OpDesc = {
  name: string,
  inferShape: (inputs : Array<Shape>) => Shape,
  gradient: Function
};

export class Op {
  id: string;
  type: string;
  inputs: Array<Tensor>;
  outputs: Array<Tensor>;
  shape: Shape;
  visited : boolean;
  result : ndarray;

  constructor(type : string, id : string, attrs : any = {}) {
    this.type = type;
    this.outputs = [];
    this.id = id;

    const {
      inputs = []
    } = attrs;

    for(let t of inputs) {
      t.outputs.push(this);
    }
    this.inputs = inputs;
  }
  
  getShape() : Shape {
    return this.shape;
  }

  addInput(t : Tensor) {
    this.inputs.push(t);
  }

  compute(inputs : Array<ndarray>) {
    console.log(`Computing ${this.id}`);
    return inputs[0];
  }
}

export class MatMul extends Op {
  constructor(id : string, attrs: any = {}){
    super('MatMul', id, attrs);
  }

  getShape (inputs? : Array<Shape>) {
    const shapes = inputs || this.inputs.map(t => t.getShape());
    if (shapes[0][1] !== shapes[1][0] ) {
      throw `incompatible tensor shapes for matmul ${shapes[0][1]} and ${shapes[1][0]}`;
    }
    return [shapes[0][0], shapes[1][shapes[1].length - 1]]
  }
    
  compute(inputs : Array<ndarray>){
    let y = ndarray([], this.getShape(inputs.map(t => t.shape)));
    gemm(y, inputs[0], inputs[1]);
    this.result = y;
    return this.result;
  }

  gradient () {
  }
}

export class Plus extends Op {
  constructor(id : string, attrs : any = {}){
    super('Plus', id, attrs);
  }

  getShape (inputs? : Array<Shape>) {
    const shapes = inputs || this.inputs.map(t => t.getShape());
    for(let i = 0; i < shapes.length; i++) {
      if(shapes[0][i] !== shapes[1][i])
        throw `incompatible dimension ${i}: expect ${shapes[0][i]} to equal ${shapes[1][i]}`;
    }
    return shapes[0];
  }

  gradient () {
  }
}

export class Variable extends Op {
  constructor(id: string, attrs : any = {}) {
    super('Variable', id);
    const {
      shape = []
    } = attrs;
    this.shape = shape;
  }
}

export class Input extends Op {
  constructor(id : string, attrs : any = {}) {
    super('Input', id);
    const {
      shape = []
    } = attrs;
    this.shape = shape;
  }
}
