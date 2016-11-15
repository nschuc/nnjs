import test from 'ava';
import ad from '../src/ad';

test('ad on f(x) = 2*x', t => {
  const f1 = (x) => 2*x;
  const dual_f = ad(f1);
  let df = dual_f([1, 1]);
  t.is(df[0], f1(1), 'Function value still works');
  t.is(df[1], 2, 'Derivative is properly computed');
})

test('ad on f(x) = 2*x*x + 1', t => {
  const f = (x) => 2*x*x + x;
  const df = (x) => 4*x + 1;

  const x = 3;
  let dual = ad(f, x)([x, 1]);
  t.is(dual[0], f(x), 'Function value still works');
  t.is(dual[1], df(x), 'Derivative is properly computed');
})
