// @flow
import ndarray from 'ndarray';
import { Op, Input, MatMul, Plus, Variable } from './ops';
import Tensor from './tensor';
import type { Shape } from './tensor';


export default class Graph {
  static defaultGraph;
  nodes : { [id : string]: Op};
  typeCounts : { [id : string]: number};

  constructor(){
    this.typeCounts = {};
    this.nodes = {};
    Graph.defaultGraph = Graph.defaultGraph || this;
  }

  async compute( inputData : any, outputs : any ) {
    // Map inputData to input nodes
    const inputs = Object.keys(this.nodes)
                     .map(n => this.nodes[n])
                     .filter(op => !op.inputs.length);
    for(let input of inputs) {
      input.result = inputData[input.id];
    }
    await this.traverse(inputs);
  }

  async traverse(nodes : Array<Op>) {
    if(!nodes.length) return; // done
    if(nodes.length > 1) {
      await Promise.all(nodes.map(n => this.traverse([ n ])))
      return 
    }
    else {
      const node = nodes[0];

      if(node.type === 'Input') return;
      if(node.visited) return;
      const dependencies = node.inputs.map(t => t.input.result).filter(a => a);
      if(dependencies.length !== node.inputs.length) return;

      const res = node.compute(dependencies);
      node.visited = true;

      let outOps : Array<Op> = [];
      node.outputs.forEach(t => {
        outOps = [...outOps, ...t.outputs ];
      });

      await this.traverse(outOps);
    }
  }

  createId(type : string) : string {
    const counts = this.typeCounts;
    counts[type] = (counts[type] || 0) + 1;
    return `${type}_${counts[type]}`;
  }

  /*
   * Creates a new variable and returns a
   * wrapper containing the graph node and
   * and instance of the graph containing it.
   *
   * If `name` param is defined and exists in the graph
   * then the corresponding node is wrapped and returned.
   */
  variable(shape : Shape, name? : string) {
    return this.use('var')({ shape }, name);
  }

  input(shape : Shape, name? : string) {
    return this.use('input')({ shape }, name);
  }

  use( opName: string ) {
    return (attr : any, name? : string) => {
      const id = name || this.createId(opName);
      let op;
      switch(opName) {
        case 'mm':
          op = new MatMul(id, attr);
          break;
        case 'plus':
          op = new Plus(id, attr);
          break;
        case 'var':
          op = new Variable(id, attr);
          break;
        case 'input':
          op = new Input(id, attr);
          break;
      }
      if(!op) throw `${opName} is not a valid operation`;
      // Test that op shape is valid
      op.getShape();
      this.nodes[op.id] = op;
      return new Tensor(op, this);
    }
  }
}
