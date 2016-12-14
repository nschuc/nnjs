import test from 'ava';
import Graph from '../src/graph';
import Tensor from '../src/tensor';
import { closeEnough } from '../src/utils';
import ndarray from 'ndarray';

import {
  MatMul,
  Plus,
  Sub,
  ReduceSum
} from '../src/ops';

const reduceData = {
  W: ndarray(new Float32Array([1, -2, 1, 3, 5, 12]), [3,2]),
  y: ndarray(new Float32Array([-1, 4, 17]), [3])
}

const mmData = {
  W: ndarray(new Float32Array([0, -1, 1, 0, 4.7, 12]), [3,2]),
  x: ndarray(new Float32Array([1, 2.3, 1, 1, 3, 4]), [2,3]),
  y: ndarray(new Float32Array([-1, -3, -4, 1, 2.3, 1, 16.7, 46.81, 52.7]), [3,3])
}

const matTestData = {
  a: ndarray(new Float32Array([0, -1, 1, 0, 4.7, 12]), [3,2]),
  b: ndarray(new Float32Array([1, 2.3, 1, 1, 3, 4]), [3,2]),
  a_plus_b: ndarray(new Float32Array([1, 1.3, 2, 1, 7.7, 16]), [3,2]),
  a_sub_b: ndarray(new Float32Array([-1, -3.3, 0, -1, 1.7, 8]), [3,2])
}


test('tensor shape gets set', t => {
  const G = new Graph()
  const t1 = G.variable([5, 3, 4])
  t.deepEqual(t1.getShape(), [5, 3, 4], 'does not get shape set');
})

test('tensor shapes are computed properly', t => {
  const G = new Graph()

  const W = G.variable([50, 30], 'W');
  const b = G.variable([50, 1], 'b');
  const x = G.input([30, 1], 'x');

  let result = W.mm(x);
  t.deepEqual(result.getShape(), [50, 1], 'result of matrix mult is not right');
  
  result = result.plus(b);
  t.deepEqual(result.getShape(), [50, 1], 'result of matrix addition is not right');
})

test('matrix multiplication computation', t => {
  const op = new MatMul();
  const y = op.compute([mmData.W, mmData.x]);
  t.true(closeEnough(y, mmData.y), 'result is not close enough');
  t.deepEqual(y.shape, mmData.y.shape, 'shape is wrong');
})

test('matrix addition computation', t => {
  const op = new Plus();
  const y = op.compute([matTestData.a, matTestData.b]);
  t.true(closeEnough(y, matTestData.a_plus_b), 'result is not close enough');
  t.deepEqual(y.shape, matTestData.a_plus_b.shape, 'shape is wrong');
})

test('matrix subtraction computation', t => {
  const op = new Sub();
  const y = op.compute([matTestData.a, matTestData.b]);
  t.true(closeEnough(y, matTestData.a_sub_b), 'result is not close enough');
  t.deepEqual(y.shape, matTestData.a_sub_b.shape, 'shape is wrong');
})

test('reduce sum computation', t => {
  const op = new ReduceSum('', {dim: 1});
  t.deepEqual(op.inferShape([reduceData.W.shape]), reduceData.y.shape, 'shape is wrong');
  const y = op.compute([reduceData.W]);
  t.true(closeEnough(y, reduceData.y), 'result is not close enough');
  t.deepEqual(y.shape, reduceData.y.shape, 'shape is wrong');
})

test('incompatible shapes throws an error', t => {
  const G = new Graph()
  const W = G.variable([50, 30], 'W');
  const b = G.variable([50, 1], 'b');

  t.throws(() => {
    const result = W.mm(b);
  });

  t.throws(() => {
    const result = W.plus(b);
  });
});
