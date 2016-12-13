// @flow
import Graph from './graph';
import type { Shape } from './tensor';

import ndarray from 'ndarray';
import cwise from 'cwise';
import zerose from 'cwise';
import cpuops from 'ndarray-ops';
import gemm from 'ndarray-gemm';

export class Op {
  id: string;
  type: string;
  inputs: Array<Op>;
  outputs: Array<Op>;
  shape: Shape;
  visited : boolean;
  result : ndarray;
  depCount : number;

  constructor(type : string, id : string, attrs : any = {}) {
    this.type = type;
    this.outputs = [];
    this.id = id;

    const {
      shape,
      inputs = []
    } = attrs;

    this.inputs = inputs.map(t => t.input);
    if(shape) {
      this.shape = shape;
    }
    else {
      const inShapes = this.inputs.map(t => t.shape);
      if(inShapes.length)
        this.shape = this.inferShape(inShapes);
    }

    for(let t of inputs) {
      t.input.outputs.push(this);
      this.depCount++;
    }
  }

  inferShape(shapes : Array<Shape>) : Shape {
    return this.shape;
  }

  compute(inputs : Array<ndarray>) {
    return inputs[0];
  }
}

export class MatMul extends Op {
  constructor(id : string, attrs: any = {}){
    super('MatMul', id, attrs);
  }

  inferShape (shapes : Array<Shape>) :Shape {
    if ( shapes[0][1] !== shapes[1][0] ) {
      throw `incompatible tensor shapes for matmul ${shapes[0][1]} and ${shapes[1][0]}`;
    }
    return [shapes[0][0], shapes[1][1]];
  }

  compute(inputs : Array<ndarray>){
    const shape = this.inferShape(inputs.map(a => a.shape));
    let y = ndarray([], shape);
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

  inferShape (shapes : Array<Shape>) {
    for(let i = 0; i < shapes.length; i++) {
      if(shapes[0][i] !== shapes[1][i])
        throw `incompatible dimension ${i}: expect ${shapes[0][i]} to equal ${shapes[1][i]}`;
    }
    return shapes[0];
  }

  compute(inputs : Array<ndarray>){
    const shape = this.inferShape(inputs.map(a => a.shape));
    let y = ndarray([], shape);
    cpuops.add(y, inputs[0], inputs[1]);
    this.result = y;
    return this.result;
  }

  gradient () {
  }
}

export class Variable extends Op {
  constructor(id: string, attrs : any = {}) {
    super('Variable', id, attrs);
    const {
      shape = []
    } = attrs;
    this.shape = shape;
  }

  compute() {
    return this.result;
  }
}

export class Input extends Op {
  constructor(id : string, attrs : any = {}) {
    super('Input', id, attrs);
    const {
      shape = []
    } = attrs;
  }
  compute() {
    return this.result;
  }
}
