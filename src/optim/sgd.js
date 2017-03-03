import Variable from "../autodiff";

export default class SGD {
  params: Array<Variable>;
  lr: number;
  constructor(params: Array<Variable>, { lr }: any) {
    if (!lr) {
      throw "Learning Rate required";
    }
    this.lr = lr;
    this.params = params;
  }

  step() {
    for (let p of this.params) {
      p.data = p.data.sub(p.grad.mul(this.lr));
    }
  }

  zero_grad() {
    for (let p of this.params) {
      if (p.grad) {
        p.grad.zero_();
      }
    }
  }
}
