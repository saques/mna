function v = eigtrivshifthess(a, tolerance, dim)
  x = hess(a);
  v = [];
  d = dim;
  for i = 1:d
    id = eye(dim);
    do
      prev = x(dim,dim);
      mu = prev;
      [q,r] = qr(x-mu*id);
      x = r*q + mu*id;
    until(abs(prev-x(dim,dim))<tolerance)
    v(end+1)=x(dim,dim);
    dim--;
    x = x(1:dim, 1:dim);
  end
end