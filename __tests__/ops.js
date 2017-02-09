import nj from 'numjs';

import Tensor from '../src/tensor.js';
import nn from '../';
import * as ops from '../src/autodiff/ops.js';
import { closeEnough, matTestData, mmData } from './data.js';

describe('Add Op', () => {
  it('should add two tensors', () => {
    let op = new ops.Add();
    let t1 = new Tensor({ data: matTestData.a });
    let t2 = new Tensor({ data: matTestData.b });
    let v3 = op.forward(t1, t2);
    const res = v3.numjs();
    expect(closeEnough(res, matTestData.a_plus_b)).toBe(true);
  });

  it('should compute proper gradient', () => {
    let op = new ops.Add();
    let t = nn.randn(3, 3);
    let grads = op.backward(t);
    expect(closeEnough(grads[0].numjs(), t.numjs())).toBe(true);
    expect(closeEnough(grads[1].numjs(), t.numjs())).toBe(true);
  });
})

describe('MatMul Op', () => {
  it('forward should add multiply matrices', () => {
    let op = new ops.MatMul();
    let t1 = new Tensor({ data: mmData.W });
    let t2 = new Tensor({ data: mmData.x });
    let v3 = op.forward(t1, t2);
    const res = v3.numjs();
    expect(closeEnough(res, mmData.y)).toBe(true);
  });

  it('should compute proper gradient', () => {
    let op = new ops.MatMul();
    let a = new Tensor({
      data: nj.array([[-0.16766106,  0.39275593],
                [-0.26878762,  0.18924427],
                [ 0.86349499, -0.27570254]])
    });
    let b = new Tensor({
      data: nj.array([[-0.68068945,  0.66590494,  1.89313304],
                [-1.22799885,  0.54066521,  0.33399853]])
    });
    let c = new Tensor({
      data: nj.array([[-0.36817873,  0.10070314, -0.18622479],
                [-0.04943085, -0.07666922, -0.44564345],
                [-0.24920955,  0.42594284,  1.54262662]])
    });
    const fwd = op.forward(a, b);
    let grads = op.backward(nn.ones(3, 3));

    let a_grad = new Tensor({
			data: nj.array([[ 1.87834859, -0.35333511],
								[ 1.87834859, -0.35333511],
								[ 1.87834859, -0.35333511]])
		});

    let b_grad = new Tensor({
			data: nj.array([[ 0.4270463 ,  0.4270463 ,  0.4270463 ],
								[ 0.30629766,  0.30629766,  0.30629766]])
		});


    expect(closeEnough(fwd.numjs(), c.numjs())).toBe(true);
    expect(closeEnough(grads[0].numjs(), a_grad.numjs())).toBe(true);
    expect(closeEnough(grads[1].numjs(), b_grad.numjs())).toBe(true);
  });
})