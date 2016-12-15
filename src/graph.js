// @flow
import ndarray from 'ndarray';
import { 
  Op, Input, Variable, MatMul, Plus, Sub, Pow, ReduceSum
} from './ops';
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

   compute( inputData : any, outputs : any ) {
    const inputs = this.feedInputs(inputData);
    this.traverse(inputs, node => {
      if(node.visited) return [];
      if(node.depCount > 0) return []; // dependencies aren't finished
      const dependencies = node.inputs
        .map(op => op.result)
        .filter(a => a);
      const res = node.compute(dependencies);
      node.outputs.forEach(node => node.depCount--);
      node.visited = true;
      return node.outputs;
    });

    const results = this.getResults(outputs);
    return results;
  }


  async traverse(nodes : Array<Op>, visitor : (node : Op) => Array<Op>) {
    if(!nodes.length) return; // done
    if(nodes.length > 1) {
      await Promise.all(nodes.map(n => this.traverse([ n ], visitor)));
      return;
    }
    else {
      const next = visitor(nodes[0]);
      await this.traverse(next, visitor);
    }
  }

  createId(type : string) : string {
    const counts = this.typeCounts;
    counts[type] = (counts[type] || 0) + 1;
    return `${type}_${counts[type]}`;
  }

  feedInputs(inputData : { [id: string] : ndarray }) : Array<Op> {
    // Map inputData to input nodes
    const inputs = Object.keys(this.nodes)
                     .map(n => this.nodes[n])
                     .filter(op => !op.inputs.length);

    if(Object.keys(inputData).length < inputs.length) {
      throw 'Not enough inputs provided to perform computation';
    }

    for(let input of inputs) {
      if(!inputData[input.id]) throw `Input data not found for '${input.id}'`;
      input.result = inputData[input.id];
    }

    return inputs;
  }

  getResults(outputs: { [id:string] : Tensor }) {
    let results = {};
    for(const key in outputs) {
      results = {
        ...outputs,
        [key] : outputs[key].input.result
      }
    }
    return results;
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
    return (attrs : any, name? : string) => {
      const id = name || this.createId(opName);
      let op;
      switch(opName) {
        case 'mm':
          op = new MatMul(id, attrs);
          break;
        case 'plus':
          op = new Plus(id, attrs);
          break;
        case 'sub':
          op = new Sub(id, attrs);
          break;
        case 'pow':
          console.log(attrs);
          op = new Pow(id, attrs);
          break;
        case 'reduce_sum':
          op = new ReduceSum(id, attrs);
          break;
        case 'var':
          op = new Variable(id, attrs);
          break;
        case 'input':
          op = new Input(id, attrs);
          break;
      }
      if(!op) throw `${opName} is not a valid operation`;

      this.nodes[op.id] = op;

      return new Tensor(op, this);
    }
  }
}
