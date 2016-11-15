import fwd from './fwd';

const ad = (fn) => {
  const dual_fn = fwd(fn);
  return dual_fn;
}

export default ad
