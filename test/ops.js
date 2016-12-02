import test from 'ava';
import Graph from '../src/graph';
import { MatMul } from '../src/ops';
import Tensor from '../src/tensor';
import ndarray from 'ndarray';

const testData = {
  W: ndarray(new Float32Array([0, -1, 1, 0, 4.7, 12]), [3,2]),
  x: ndarray(new Float32Array([1, 2.3, 1, 1, 3, 4]), [2,3]),
  y: ndarray(new Float32Array([-1, -3, -4, 1, 2.3, 1, 16.7, 46.81, 52.7]), [3,3])

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

test.only('matmul computation works', t => {
  const op = new MatMul();
  const y = op.compute([testData.W, testData.x]);
  const closeEnough = (a, b, eps=1e-3) => {
    for(let i = 0; i < a.data.length; i++) {
      console.log(a.data[i], b.data[i]);
      if(Math.abs(a.data[i] - b.data[i]) > eps) return false;
    }
    return true;
  }
  t.true(closeEnough(y, testData.y), 'result is not close enough');
  t.deepEqual(y.shape, testData.y.shape, 'shape is wrong');
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
