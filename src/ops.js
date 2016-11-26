// @flow
//
import Tensor from './graph';
import type { Shape } from './graph';

export type OpDesc = {
  name: string,
  inferShape: (inputs : Array<Shape>) => Shape,
  gradient: Function
};

const mm : OpDesc = {
  name: 'matmul',

  inferShape: (inputs : Array<Shape>) => {
    if (inputs[0][1] !== inputs[1][0] ) {
      throw `incompatible tensor shapes for matmul ${inputs[0][1]} and ${inputs[1][0]}`;
    }
    return [inputs[0][0], inputs[1][inputs[1].length - 1]]
  },

  gradient: ({ inputs, output }) => {
    return [
      inputs[1].mm(output), 
      inputs[0].mm(output)
    ]
  }
}

const plus : OpDesc = {
  name: 'plus',

  inferShape: (inputs : Array<Shape>) => {
    if(!inputs.length) throw "Not enough inputs";
    for(let i = 0; i < inputs.length; i++) {
      if(inputs[0][i] !== inputs[1][i])
        throw `incompatible dimension ${i}: expect ${inputs[0][i]} to equal ${inputs[1][i]}`;
    }
    return inputs[0];
  },

  gradient: (inputs) => {
    return [
      inputs[1], 
      inputs[0]
    ]
  }
}

export default {
  mm,
  plus
}
