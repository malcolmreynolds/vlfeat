import numpy as np
cimport numpy as np

cimport cython

@cython.boundscheck(False)
def match(np.ndarray[np.uint8_t, ndim=2] desc_1, 
          np.ndarray[np.uint8_t, ndim=2] desc_2,
          float thresh=1.5):
    """Match two sets of SIFT descriptors using David Lowe's algorithm

    REFERENCES::
     [1] D. G. Lowe, Distinctive image features from scale-invariant
     keypoints. IJCV, vol. 2, no. 60, pp. 91-110, 2004."""

    # if len(desc_1.shape) != 2:
    #     raise ValueError('desc_1 must be 2D array')
    # if len(desc_2.shape) != 2:
    #     raise ValueError('desc_2 must be 2D array')

    cdef np.ndarray[np.float_t, ndim=2] desc_1_f = desc_1.astype(np.float)
    cdef np.ndarray[np.float_t, ndim=2] desc_2_f = desc_2.astype(np.float)

    cdef int k1
    cdef int k2

    cdef int num_bins_1 = desc_1.shape[0]
    cdef int num_desc_1 = desc_1.shape[1]
    
    cdef int num_bins_2 = desc_2.shape[0]
    cdef int num_desc_2 = desc_2.shape[1]

    cdef int num_pairs = num_desc_1 + num_desc_2

    if num_bins_1 != num_bins_2:
        raise ValueError('desc_1 and desc_2 must have same number of rows')

    # The vlfeat code I'm porting has this array of size num_desc_1
    # + num_desc_2 but as far as I can see it will only ever fill
    # up the first one
    cdef np.ndarray[np.int_t, ndim=2] pair_indices = -1 * np.ones((2, num_pairs), dtype=np.int)
    cdef np.ndarray[np.float32_t, ndim=1] pair_scores = np.zeros(num_pairs, dtype=np.float32)
    cdef int p_idx = 0
    cdef float best
    cdef float second_best
    cdef int best_match_idx
    cdef int bin_idx
    cdef float acc_sq
    cdef float diff

    # Compare all pairs
    for k1 in range(num_desc_1):
        # print "k1 = %d" % k1

        # We keep both the best match and the second best one,
        # to see that the best match we have is far enough away
        # from the second best one (otherwise things are a bit ambiguous)
        best = np.inf
        second_best = np.inf
        best_match_idx = -1

        for k2 in range(num_desc_2):
            # compute squared difference between descriptors k1 and k2
            acc_sq = 0.0
            for bin_idx in range(num_bins_1):
                diff = desc_1_f[bin_idx, k1] - desc_2_f[bin_idx, k2]
                acc_sq += (diff * diff)

            if acc_sq < best:
                second_best, best, best_match_idx = best, acc_sq, k2
            elif acc_sq < second_best:
                second_best = acc_sq

        if (thresh * best) < second_best and best_match_idx != -1:
            # Store the match and increment the output counter
            # print "matched k1 %d to k2 %d" % (k1, best_match_idx)
            pair_indices[0, p_idx] = k1
            pair_indices[1, p_idx] = best_match_idx
            # pair_indices[:, p_idx] = [k1, best_match_idx]
            pair_scores[p_idx] = best
            p_idx += 1

    # Return results
    return pair_indices[:, :p_idx], pair_scores[:p_idx]
