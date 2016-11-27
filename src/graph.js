// @flow
import opDescriptions from './ops';
import ndarray from 'ndarray';

export type Shape = Array<number>

type TargetDict = { [id:string]: Wrapper };
type FeedDict = { [id:string]: ndarray };

export default class Graph {
  nodes : Map<string, Node>;
  typeCounts : Map<string, number>;

  constructor(){
    this.typeCounts = new Map();
    this.nodes = new Map();
  }
 
  addOp(type : string, deps : Array<Node>) : Node {
    const opDesc = opDescriptions[type];
    const shape = opDesc.inferShape(deps.map(t => t.getShape()));
    
    const id = this.createId(type);
    const node = new Node(id, type, shape);
    deps.forEach( d => {
      d.addOutput(node.id);
      node.addInput(d.id);
    });

    this.nodes.set(node.id, node);
    return node;
  }

  createId(type : string) : string {
    const count = this.typeCounts.get(type) || 0;
    this.typeCounts.set(type, count+1);
    return `${type}_${count}`;
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
    if(name) {
      const node = this.nodes.get(name);
      if(node) return new Wrapper(node, this);
    }
    return this.createAndWrapNode('variable', shape, name);
  }

  input(shape : Shape, name? : string) {
    return this.createAndWrapNode('input', shape, name);
  }

  createAndWrapNode(type : string, shape : Shape, name? : string) {
    const id = name || this.createId(type);
    const node = new Node(id, type, shape);
    this.nodes.set(node.id, node);
    return new Wrapper(node, this);
  }

}

const topologicalSort = (startNodes : Array<string>, graph: Map<string, Node>) => {
  let sorted = [];
  let visited = new Set();
  let queue = startNodes;
  while(queue.length) {
    const front = queue.shift();
    if(!visited.has(front)) { 
      visited.add(front)
      sorted.push(front);
      const frontNode = graph.get(front);
      if(frontNode) {
        queue = queue.concat(frontNode.inputs);
      }
    }
  }
  return sorted;
}

class Node {
  id: string;
  type: string;
  inputs: Array<string>;
  outputs: Array<string>;
  shape: Shape;

  constructor(id : string, type : string, shape : Shape) {
    this.id = id;
    this.type = type;
    this.shape = shape;
    this.inputs = [];
    this.outputs = [];
  }
  
  getShape() : Shape {
    return this.shape;
  }

  addInput(id : string) {
    this.inputs.push(id);
  }

  addOutput(id : string) {
    this.outputs.push(id);
  }
}

export class Wrapper {
  node: Node;
  graph: Graph;

  constructor(node: Node, graph : Graph) {
    this.node = node;
    this.graph = graph;
  }

  getShape() : Shape {
    return this.node.getShape();
  }

  getId() : string {
    return this.node.id;
  }

  mm(t2 : Wrapper) : Wrapper {
    const newNode = this.graph.addOp('mm', [this.node, t2.node]);
    return new Wrapper(newNode, this.graph);
  }

  plus(t2 : Wrapper) : Wrapper {
    const newNode = this.graph.addOp('plus', [this.node, t2.node]);
    return new Wrapper(newNode, this.graph);
  }
}
