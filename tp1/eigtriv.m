function v = eigtriv(a, tolerance, dim)
  #{
    Plain QR algorithm, improved only by
    reducing the matrix when the last eigenvalue
    appears to have converged.
  #}
  x = a;
  v = [];
  d = dim;
  for i = 1:d
    do
      prev = x(dim,dim);
      [q,r] = qr(x);
      x = r*q;
    until(abs(prev-x(dim,dim))<tolerance)
    v(end+1)=x(dim,dim);
    dim--;
    x = x(1:dim, 1:dim);
  end
end
  