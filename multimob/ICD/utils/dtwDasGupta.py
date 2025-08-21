import numpy as np
from numba import njit

@njit
def dtwdasgupta(r, t):
    """
      Fast DTW-based warping function.

      This function aligns the second time series `t` to the first time series `r`
      using Dynamic Time Warping (DTW), and returns a warped version of `t` that
      follows the temporal structure of `r`. The warping path is averaged when
      multiple values from `t` map to the same index in `r`.

      Parameters
      ----------
      r : np.ndarray
          Reference time series (e.g., yaw1).
      t : np.ndarray
          Target time series to be warped (e.g., yaw2).

      Returns
      -------
      np.ndarray
          Warped version of `t`, aligned to `r` in time.

      Notes
      -----
      - This version is ~300× faster than a naïve Python implementation.
      - It uses Numba's `@njit` for just-in-time compilation.
      - Multiple mappings to the same index are averaged to preserve signal energy.
      """

    M = len(r)
    N = len(t)

    # Compute squared differences
    d = np.empty((M, N))
    for i in range(M):
        for j in range(N):
            diff = r[i] - t[j]
            d[i, j] = diff * diff

    # Initialize the DTW cost matrix
    D = np.empty((M, N))
    D[0, 0] = d[0, 0]

    for m in range(1, M):
        D[m, 0] = d[m, 0] + D[m - 1, 0]
    for n in range(1, N):
        D[0, n] = d[0, n] + D[0, n - 1]

    for m in range(1, M):
        for n in range(1, N):
            D[m, n] = d[m, n] + min(D[m - 1, n], D[m, n - 1], D[m - 1, n - 1])

    # Backtrack to find the warping path
    m = M - 1
    n = N - 1
    max_len = M + N
    path_p = np.empty(max_len, dtype=np.int32)
    path_q = np.empty(max_len, dtype=np.int32)
    k = 0

    while m > 0 or n > 0:
        path_p[k] = m
        path_q[k] = n
        k += 1

        if m == 0:
            n -= 1
        elif n == 0:
            m -= 1
        else:
            cost = np.array([D[m - 1, n], D[m, n - 1], D[m - 1, n - 1]])
            argmin = np.argmin(cost)
            if argmin == 0:
                m -= 1
            elif argmin == 1:
                n -= 1
            else:
                m -= 1
                n -= 1

    path_p[k] = 0
    path_q[k] = 0
    k += 1

    # Reverse path
    out_len = np.max(path_p[:k]) + 1
    yaw2new = np.zeros(out_len)
    count = np.zeros(out_len)

    for i in range(k - 1, -1, -1):  # Reverse order
        pi = path_p[i]
        qi = path_q[i]
        yaw2new[pi] += t[qi]
        count[pi] += 1

    # Average overlapping assignments
    for i in range(out_len):
        if count[i] > 0:
            yaw2new[i] /= count[i]

    return yaw2new
