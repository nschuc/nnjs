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
})
