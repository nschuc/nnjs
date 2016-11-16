import * as esprima from 'esprima';
import * as escodegen from 'escodegen';

import { traverse, createCallSite, injectOps, getId } from './utils';

const addToGraph = (node, graph) => {
  const id = getId(graph, node.operator);
  graph[id] = {
    op: node.operator,
    id,
    in_edges: [
      node.left.name, node.right.name
    ],
    out_edges: [],
  }
  graph[node.left.name].out_edges.push(id)
  graph[node.right.name].out_edges.push(id)
  return {
    name: id
  }
}

const getInputs = (graph) =>
  Object.keys(graph).filter(key => graph[key].in_edges.length === 0);

const getOutputs = (graph) =>
  Object.keys(graph).filter(key => graph[key].out_edges.length === 0);

const topologicalSort = (graph) => {
  let sorted = [];
  let visited = new Set();
  const outputs = getOutputs(graph);
  let queue = outputs;
  while(queue.length) {
    const front = queue.shift();
    if(!visited.has(front)) { 
      visited.add(front)
      sorted.push(front);
      queue = queue.concat(graph[front].in_edges);
    }
  }
  return sorted;
}

const opFuncs = {
  'input': () => {},
  '*': (a, b) => a * b,
  '/': (a, b) => a / b,
  '+': (a, b) => a + b,
  '-': (a, b) => a - b
}

const dfFuncs = {
  'input': () => {},
  '*': (deps, wrt) => {
    if(deps.every(d => d.id === wrt)) {
      return deps[0].val;
    }
    return deps
      .filter(d => d.id !== wrt)
      .map(d => d.val)
      .reduce((a, b) => a+b, 0);
  },
  '+': (deps, wrt) => {
    if(deps.every(d => d.id === wrt)) {
      return 1;
    }
    return deps
      .filter(d => d.id === wrt)
      .map(a => 1)
      .reduce((a, b) => a+b, 0);
  },
  '-': (a, b) => {
    if(deps.every(d => d.id === wrt)) {
      return -1;
    }
    return deps
      .filter(d => d.id === wrt)
      .map(a => -1)
      .reduce((a, b) => a+b, 0);
  },

}

const execGraphFwd = (order, ops, graph) => {
  order.forEach( id => {
    const op = graph[id].op;
    const operands = graph[id].in_edges.map(id => graph[id].val);
    graph[id].val = graph[id].val || ops[op](...operands);
  });
}

const execGraphRev = (order, ops, graph) => {
  order.forEach( i => {
    graph[i].in_edges.forEach( j => {
      graph[j].dv = graph[j].dv || 0; // initialize

      const { op, dv} = graph[i];
      const didj = dfFuncs[op](
          graph[i].in_edges.map(i => graph[i]), j
          );
      graph[j].dv += didj * graph[i].dv;
    });
  });
}

const rev = (fn) => {
  if( typeof fn !== 'function' ) {
    throw 'nice try buddy'
  }

  // Construct ast
  const fnString = fn.toString();
  var ast = esprima.parse(fnString);

  let graph = {};
  const params = ast.body[0].params.map(i => i.name);

  params.forEach(p => {
    graph[p] = { op: 'input', id: p, in_edges: [], out_edges: [] }
  });

  traverse(ast, {
    BinaryExpression: addToGraph
  }, graph);

  const opOrdering = topologicalSort(graph).reverse();

  return (...inValues) => {
    let inputs = {}
    for(const i in params){
      graph[params[i]].val = inValues[i];
    }
    execGraphFwd(opOrdering, opFuncs, graph);

    const outputs = getOutputs(graph);
    outputs.forEach( o => graph[o].dv = 1 )
    execGraphRev(opOrdering.reverse(), dfFuncs, graph);
    return params.map(p => graph[p].dv);
  };
}

export default rev
