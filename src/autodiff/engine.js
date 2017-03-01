import Variable from "./variable.js";

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

export { differentiate };
