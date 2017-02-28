import Variable from "./variable.js";

const topological = (vars: Array<Variable>) => {
  let sorted = [];
  let visited = new Set();
  let queue = vars;
  while (queue.length) {
    const front = queue.shift();
    if (!visited.has(front)) {
      visited.add(front);
      sorted.push(front);
      queue = queue.concat(graph[front].in_edges);
    }
  }
  return sorted;
};

const differentiate = (vars: Array<Variable>, grad: Tensor) => {
  for (let v of vars) {
    v.grad = v.grad ? v.grad.add_(grad) : grad;
    if (v.creator) {
      const parents = v.creator.inputs;
      const grads = v.creator.backward(grad);
      for (let i = 0; i < parents.length; ++i) {
        differentiate([parents[i]], grads[i]);
      }
    }
  }
};

class AutoDiffEngine {
  diff(vars: Array<Variable>, grad: Tensor) {
  }
}

export { differentiate };
