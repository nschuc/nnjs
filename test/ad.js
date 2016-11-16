import test from 'ava';
import { fwd, rev } from '../src/ad';

test('ad on f(x) = 2*x', t => {
  const f1 = (x) => 2*x;
  const dual_f = fwd(f1);
  let df = dual_f([1, 1]);
  t.is(df[0], f1(1), 'Function value still works');
  t.is(df[1], 2, 'Derivative is properly computed');
})

test('ad on f(x) = 2*x*x + 1', t => {
  const f = (x) => 2*x*x + x;
  const df = (x) => 4*x + 1;

  const x = 3;
  let dual = fwd(f, x)([x, 1]);
  t.is(dual[0], f(x), 'Function value still works');
  t.is(dual[1], df(x), 'Derivative is properly computed');
})

test('reverse mode f(x, y) = x^2 + y^2', t => {
  const f = (x, y) => x*x + y*y;
  const df = (x, y) => [2*x, 2*y];

  const x = 3, y = 4;
  let res = rev(f)(x, y);
  t.deepEqual(res, df(x, y), 'Derivative is properly computed');
})

test('reverse mode f(x, y)', t => {
  const f = (x, y) => x*x*y + y*y;
  const df = (x, y) => [2*x*y, x*x + 2*y];

  const x = 3, y = 4;
  let res = rev(f)(x, y);
  t.deepEqual(res, df(x, y), 'Derivative is properly computed');
})
