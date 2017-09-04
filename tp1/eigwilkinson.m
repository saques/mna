function v = eigwilkinson(a, tolerance, dim)
  #{
    QR Algorithm for computing eigenvalues,
    starting from the Hessemberg form of a given
    matrix and performing wilkinson shifts at each
    step.
  #}
  x = hess(a);
  v = [];
  d = dim;
  for i = 1:d-1
    id = eye(dim);
    do
      prev = x(dim,dim);
      mu = wilkinson(x(dim-1, dim-1), x(dim-1, dim), x(dim, dim));
      [q,r] = qr(x - mu*id);
      x = r*q + mu*id;
    until (abs(prev-x(dim,dim))<tolerance)
    
    v(end+1)=x(dim,dim);
    dim--;
    x = x(1:dim, 1:dim);
  end
end
  