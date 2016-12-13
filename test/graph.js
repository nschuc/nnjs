import test from 'ava';
import Graph from '../src/graph';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { closeEnough } from '../src/utils';

const testData = {
  W : ndarray(
          new Float32Array([ -0.04187682, -0.10992487,  0.1612422 , -1.40858478, -1.79128775,
        0.11002018, -1.82981874, -0.99152277,  0.61951961, -0.92007722,
        0.12116806, -0.68038318, -0.51290442,  0.0412081 ,  0.11992956,
       -0.07240007,  0.03892415, -0.89676515, -1.44497428,  0.56576945,
        0.46126791,  0.65752456,  0.71926312,  2.26230551,  2.83311028,
        0.19954042, -0.34248868,  1.02768874, -0.87839714,  1.15928717,
       -0.99786876, -0.23665498,  0.04692561,  0.73027127,  1.00809621,
        1.87618061,  0.10362345,  0.57414773,  0.08032668,  0.81123704,
        0.16278944, -0.9237439 , -0.91559788,  1.30627659,  1.08274194,
        0.51288325,  1.46104946,  0.53820037, -1.41988554,  0.66250847,
       -0.56647393, -1.19174359,  0.69055124, -0.50980072, -0.35467602,
        0.16135574,  0.73541336, -0.04648473, -0.20894094,  1.92703045,
       -0.86457206,  0.96868025,  1.66506422, -0.27868347,  0.24558642,
        0.57165889,  0.13108033,  1.79441055, -0.03259253, -2.33701101,
        0.76604203,  0.48507049, -0.08169726, -0.05241587, -1.46912711,
       -0.1735999 , -0.4810902 , -0.18951443,  0.53071252, -0.45393259,
        1.09280184,  1.50950009, -1.15867761,  0.06924176, -1.17560131,
       -1.1838838 ,  0.54504182, -1.84463796,  0.69850134, -1.77372341,
        0.38131482,  0.35403953, -0.95314146, -2.17358802,  1.42805239,
        1.80620735, -0.12367717, -0.64491831, -0.51480173, -1.00618475]), [10, 10]),

	x : ndarray(new Float32Array([ 0.63425948,  0.66272638,  0.58328916,  0.26560492,  0.39507165,
        0.23941119,  0.40974588,  0.28544631,  0.78777774,  0.29105603]), [10, 1]),

	b : ndarray(new Float32Array([ 0.56755259,  0.25596796,  0.38128781,  0.7410477 ,  0.588911  ,
        0.33083641,  0.77996825,  0.39013102,  0.89659157,  0.62288638]), [10, 1]),

	y : ndarray(new Float32Array([-1.30581892, -1.58994432,  3.09553427,  1.52582881,  0.26997726,
        0.03197694,  1.86469583,  0.54866495,  0.91555659,  0.02953553]), [10, 1])
}

test('graph computation works for y = Wx + b', t => {
  const G = new Graph()
  const W = G.input([10, 10], 'W');
  const b = G.input([10, 1], 'b');
  const x = G.input([10, 1], 'x');

  const y = W.mm(x).plus(b);

  const result = G.compute({
    [ x.getId() ] : testData.x,
    [ W.getId() ] : testData.W,
    [ b.getId() ] : testData.b,
  }, {
    y
  });

  t.true(closeEnough(result.y, testData.y), 'result is not close enough');
  t.is(Object.keys(G.nodes).length, 5, 'not enough ops in graph');
});

test('train a linear regression', t => {
  const G = new Graph()
  const W = G.input([10, 10], 'W');
  const b = G.input([10, 1], 'b');
  const x = G.input([10, 1], 'x');

  const y = W.mm(x).plus(b);
// Mean squared error
// cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
// optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    const result = G.compute({
    [ x.getId() ] : testData.x,
    [ W.getId() ] : testData.W,
    [ b.getId() ] : testData.b,
  }, {
    y
  });

  t.true(closeEnough(result.y, testData.y), 'result is not close enough');
  t.is(Object.keys(G.nodes).length, 5, 'not enough ops in graph');
});
