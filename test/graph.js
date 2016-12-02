import test from 'ava';
import Graph from '../src/graph';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';

test('tensor matmul builds graph properly', t => {
  const G = new Graph()
  const W = G.variable([50, 30], 'W');
  const b = G.variable([50, 1], 'b');
  const x = G.input([30, 1], 'x');

  const X_train = ndarray(new Float64Array( 30 * 1 ), [30, 1]);

  const y = W.mm(x).plus(b);

  const res = G.compute({
    [ x.getId() ] : X_train
  }, {
    y
  });

  t.is(Object.keys(G.nodes).length, 5, 'not enough ops in graph');
});
