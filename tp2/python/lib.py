import numpy as np


#Brute force implementation
def fft_bf(x):
    N = len(x)
    ans = np.empty((N,)).astype(complex)

    for k in xrange(0, N):
        s = 0 + 0j
        for n in xrange(0, N):
            s += x[n]*np.exp((-2*np.pi*1j*n*k)/N)
        ans[k] = s
    np.exp
    return ans

#Cooley-Tukey implementation (only N = 2^j for some j)
def fft_ct_1(x, N):
    ans = None
    if N == 1:
        ans = np.empty((1,)).astype(complex)
        ans[0] = x[0]
    else:
        t1 = fft_ct_1(x[::2], N/2)
        t2 = fft_ct_1(x[1::2], N/2)
        ans = np.empty((N,)).astype(complex)
        for k in xrange(0, N/2):
            t = t1[k]
            ans[k] = t + np.exp((-2*np.pi*1j*k)/N)*t2[k]
            ans[k+N/2] = t - np.exp((-2*np.pi*1j*k)/N)*t2[k]
    return ans

def fft_ct(x):
    N = len(x)
    if not (N & (N-1)) == 0:
        raise ValueError("Length must be a power of 2")
    return fft_ct_1(x, N)