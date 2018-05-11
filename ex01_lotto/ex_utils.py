import numpy as np

def shape_check(an_array):
    return np.shape(an_array)

def create_lotto_data(seq_len, seed):
    lotto = np.loadtxt('./data/lotto.csv', delimiter = ',')
    nsamples = np.size(lotto, 0)
    ndims = np.size(lotto, 1)

    for i in range(nsamples-seq_len):
        s = i
        t = s+seq_len
        tmp_x = np.expand_dims(lotto[s:t], axis=0)
        tmp_y = np.expand_dims(lotto[t], axis=0)
        if i == 0:
            lotto_x = tmp_x
            lotto_y = tmp_y
        else:
            lotto_x = np.append(lotto_x, tmp_x, axis=0)
            lotto_y = np.append(lotto_y, tmp_y, axis=0)

    print("The shape of train set : ",shape_check(lotto_x))

    num_lotto_seq = len(lotto_x)
    mask = np.random.permutation(num_lotto_seq)
    lotto_x = lotto_x[mask]
    lotto_y = lotto_y[mask]

    ntrain = int(num_lotto_seq*0.8)
    ntest = num_lotto_seq - ntrain

    lotto_train_x = lotto_x[:ntrain]/45.
    lotto_train_y = lotto_y[:ntrain]/45.
    lotto_test_x = lotto_x[-ntest:]/45.
    lotto_test_y = lotto_y[-ntest:]/45.

    return lotto_train_x, lotto_train_y, lotto_test_x, lotto_test_y