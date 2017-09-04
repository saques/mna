function x = eigenmatrix(dim, vals = [])
  #{
    Generates a random matrix with the given
    (or random) eigenvalues.
  #}
  if(size(vals) == 0 || size(vals) != dim)
    vals = rand(1,dim); 
  end
  d = diag(vals,0);
  r = rand(dim,dim);
  x = r*d*inv(r);
end
    