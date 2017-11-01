def FFT(X):
  N = len(X)
  if N > 1:
    even = X[::2]
	odd = X[1::2]
	FFT(even)
	FFT(odd)
	for k in range(N/2):
	  e = X[k]
	  o = X[k+N/2]
	  w = exp(-2*pi*1j/n)
	  X[k] = e+ w * o
	  X[k+N/2] = e – w * o
  return X