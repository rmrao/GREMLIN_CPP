# ------------------------------------------------------------
# "THE BEERWARE LICENSE" (Revision 42):
# <so@g.harvard.edu> and <pkk382@g.harvard.edu> wrote this code.
# As long as you retain this notice, you can do whatever you want
# with this stuff. If we meet someday, and you think this stuff
# is worth it, you can buy us a beer in return.
# --Sergey Ovchinnikov and Peter Koo
# ------------------------------------------------------------

# IMPORTANT, only tested using PYTHON 3!
from typing import Dict, Tuple, Optional
import contextlib
import os
import h5py
import gzip
import string
import numpy as np
import tensorflow as tf
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# ===============================================================================
# Setup the alphabet
# note: if you are modifying the alphabet
# make sure last character is "-" (gap)
# ===============================================================================
alphabet = "ARNDCQEGHILKMFPSTWYV-"
invalid_state_index = alphabet.index('-')
states = len(alphabet)
a2n: Dict[str, int] = {a: n for n, a in enumerate(alphabet)}


# ===============================================================================
# Functions for prepping the MSA (Multiple sequence alignment) from fasta/a2m file
# ===============================================================================


class SequenceLengthException(Exception):

    def __init__(self, protein_id: str):
        super().__init__("Sequence length was too long for protein {}".format(protein_id))


class TooFewValidMatchesException(Exception):

    def __init__(self, protein_id: str = None):
        message = 'There were too few valid matches'
        if protein_id is not None:
            message += ' for protein {}'.format(protein_id)
        super().__init__(message)


def to_header_and_sequence(block):
    header, *seq = block.split('\n')
    seq = ''.join(seq)
    return header, seq


def parse_fasta(filename: str, limit: int = -1, max_seq_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to parse a fasta/a2m file.

    Args:
        filename (str): filename of fasta/a2m file to load
        sequence_at_end (bool): indicates whether the actual sequence is at the beginning/end of file
        limit (int): DEPRECATED, used to limit the number of sequence matches. Need to account for
                     sequence_at_end argument to reintroduce.

    Returns:
        np.ndarray: array of headers
        np.ndarray: array of sequences
    """

    filetype = os.path.basename(filename).split('.', maxsplit=1)[1]
    assert filetype in {'a2m', 'a2m.gz', 'fasta', 'fas', 'fasta.gz', 'fas.gz'}
    is_a2m = 'a2m' in filetype
    is_compressed = 'gz' in filetype

    def get_file_obj():
        return gzip.open(filename) if is_compressed else open(filename)

    delete_lowercase_trans = ''.maketrans('', '', string.ascii_lowercase)  # type: ignore
    with get_file_obj() as f:
        fasta = f.read()
        if isinstance(fasta, bytes):
            fasta = fasta.decode()
        fasta = fasta.strip('>').translate(delete_lowercase_trans).split('>')

        if max_seq_len is not None:
            seqlen = len(to_header_and_sequence(fasta[0])[1])
            if seqlen > max_seq_len:
                raise SequenceLengthException(filename)

        if 0 < limit < len(fasta):
            headers_and_seqs = [to_header_and_sequence(block) for block in fasta[:limit]]
            if is_a2m:
                last = to_header_and_sequence(fasta[-1])
                headers_and_seqs = [last] + headers_and_seqs
        else:
            headers_and_seqs = [to_header_and_sequence(block) for block in fasta]
            if is_a2m:
                headers_and_seqs = headers_and_seqs[-1:] + headers_and_seqs[:-1]

        header, sequence = zip(*headers_and_seqs)

    return np.array(header), np.array(sequence)


def filt_gaps(msa: np.ndarray, gap_cutoff: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    '''filters alignment to remove gappy positions'''
    non_gaps = np.where(np.mean(msa == 20, 0) < gap_cutoff)[0]
    return msa[:, non_gaps], non_gaps


def get_eff(msa: np.ndarray, eff_cutoff: float = 0.8) -> np.ndarray:
    '''compute effective weight for each sequence'''
    # pairwise identity
    msa_sm = 1.0 - squareform(pdist(msa, "hamming"))

    # weight for each sequence
    msa_w = 1 / np.sum(msa_sm >= eff_cutoff, -1)

    return msa_w


def mk_msa(seqs: np.ndarray, gap_cutoff: float = 0.5):
    '''converts list of sequences to msa'''

    assert all(len(seq) == len(seqs[0]) for seq in seqs)
    msa_ori_list = [[a2n.get(aa, invalid_state_index) for aa in seq] for seq in seqs]
    msa_ori = np.array(msa_ori_list)

    # remove positions with more than > 50% gaps
    msa, v_idx = filt_gaps(msa_ori, gap_cutoff)

    if len(v_idx) == 0:
        raise TooFewValidMatchesException()

    # compute effective weight for each sequence
    msa_weights = get_eff(msa, 0.8)

    # compute effective number of sequences
    ncol = msa.shape[1]  # length of sequence
    w_idx = v_idx[np.stack(np.triu_indices(ncol, 1), -1)]

    return {"msa_ori": msa_ori,
            "msa": msa,
            "weights": msa_weights,
            "neff": np.sum(msa_weights),
            "v_idx": v_idx,
            "w_idx": w_idx,
            "nrow": msa.shape[0],
            "ncol": ncol,
            "ncol_ori": msa_ori.shape[1]}


# ===============================================================================
# GREMLIN
# ===============================================================================

def sym_w(w):
    '''symmetrize input matrix of shape (x,y,x,y)'''
    x = w.shape[0]
    w = w * np.reshape(1 - np.eye(x), (x, 1, x, 1))
    w = w + tf.transpose(w, [2, 3, 0, 1])
    return w


def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
    # adam optimizer
    # Note: this is a modified version of adam optimizer. More specifically, we replace "vt"
    # with sum(g*g) instead of (g*g). Furthmore, we find that disabling the bias correction
    # (b_fix=False) speeds up convergence for our case.

    if var_list is None:
        var_list = tf.trainable_variables()

    gradients = tf.gradients(loss, var_list)
    if b_fix:
        t = tf.Variable(0.0, "t")
    opt = []
    for n, (x, g) in enumerate(zip(var_list, gradients)):
        if g is not None:
            ini = dict(initializer=tf.zeros_initializer, trainable=False)
            mt = tf.get_variable(name + "_mt_" + str(n), shape=list(x.shape), **ini)
            vt = tf.get_variable(name + "_vt_" + str(n), shape=[], **ini)

            mt_tmp = b1 * mt + (1 - b1) * g
            vt_tmp = b2 * vt + (1 - b2) * tf.reduce_sum(tf.square(g))
            lr_tmp = lr / (tf.sqrt(vt_tmp) + 1e-8)

            if b_fix:
                lr_tmp = lr_tmp * tf.sqrt(1 - tf.pow(b2, t)) / (1 - tf.pow(b1, t))

            opt.append(x.assign_add(-lr_tmp * mt_tmp))
            opt.append(vt.assign(vt_tmp))
            opt.append(mt.assign(mt_tmp))

    if b_fix:
        opt.append(t.assign_add(1.0))
    return(tf.group(opt))


def GREMLIN(msa, opt_type="adam", opt_iter=100, opt_rate=1.0, batch_size=None):

    ##############################################################
    # SETUP COMPUTE GRAPH
    ##############################################################
    # kill any existing tensorflow graph
    tf.reset_default_graph()

    ncol = msa["ncol"]  # length of sequence

    # msa (multiple sequence alignment)
    MSA = tf.placeholder(tf.int32, shape=(None, ncol), name="msa")

    # one-hot encode msa
    OH_MSA = tf.one_hot(MSA, states)

    # msa weights
    MSA_weights = tf.placeholder(tf.float32, shape=(None,), name="msa_weights")

    # 1-body-term of the MRF
    V = tf.get_variable(name="V",
                        shape=[ncol, states],
                        initializer=tf.zeros_initializer)

    # 2-body-term of the MRF
    W = tf.get_variable(name="W",
                        shape=[ncol, states, ncol, states],
                        initializer=tf.zeros_initializer)

    # symmetrize W
    W = sym_w(W)

    def L2(x):
        return tf.reduce_sum(tf.square(x))

    ########################################
    # V + W
    ########################################
    VW = V + tf.tensordot(OH_MSA, W, 2)

    # hamiltonian
    H = tf.reduce_sum(tf.multiply(OH_MSA, VW), axis=2)
    # local Z (parition function)
    Z = tf.reduce_logsumexp(VW, axis=2)

    # Psuedo-Log-Likelihood
    PLL = tf.reduce_sum(H - Z, axis=1)

    # Regularization
    L2_V = 0.01 * L2(V)
    L2_W = 0.01 * L2(W) * 0.5 * (ncol - 1) * (states - 1)

    # loss function to minimize
    loss = -tf.reduce_sum(PLL * MSA_weights) / tf.reduce_sum(MSA_weights)
    loss = loss + (L2_V + L2_W) / msa["neff"]

    ##############################################################
    # MINIMIZE LOSS FUNCTION
    ##############################################################
    if opt_type == "adam":
        opt = opt_adam(loss, "adam", lr=opt_rate)

    # generate input/feed
    def feed(feed_all=False):
        if batch_size is None or feed_all:
            return {MSA: msa["msa"], MSA_weights: msa["weights"]}
        else:
            idx = np.random.randint(0, msa["nrow"], size=batch_size)
            return {MSA: msa["msa"][idx], MSA_weights: msa["weights"][idx]}

    # optimize!
    with tf.Session() as sess:
        # initialize variables V and W
        sess.run(tf.global_variables_initializer())

        # initialize V
        msa_cat = tf.keras.utils.to_categorical(msa["msa"], states)
        pseudo_count = 0.01 * np.log(msa["neff"])
        V_ini = np.log(np.sum(msa_cat.T * msa["weights"], -1).T + pseudo_count)
        V_ini = V_ini - np.mean(V_ini, -1, keepdims=True)
        sess.run(V.assign(V_ini))

        # compute loss across all data
        def get_loss():
            round(sess.run(loss, feed(feed_all=True)) * msa["neff"], 2)
        # print("starting", get_loss())

        if opt_type == "lbfgs":
            lbfgs = tf.contrib.opt.ScipyOptimizerInterface
            opt = lbfgs(loss, method="L-BFGS-B", options={'maxiter': opt_iter})
            opt.minimize(sess, feed(feed_all=True))

        if opt_type == "adam":
            for i in range(opt_iter):
                sess.run(opt, feed())
                # if (i + 1) % int(opt_iter / 10) == 0:
                    # print("iter", (i + 1), get_loss())

        # save the V and W parameters of the MRF
        V_ = sess.run(V)
        W_ = sess.run(W)

    # only return upper-right triangle of matrix (since it's symmetric)
    tri = np.triu_indices(ncol, 1)
    W_ = W_[tri[0], :, tri[1], :]

    mrf = {"v": V_,
           "w": W_,
           "v_idx": msa["v_idx"],
           "w_idx": msa["w_idx"]}

    return mrf


# ===============================================================================
# Explore the contact map
# ===============================================================================

# For contact prediction, the W matrix is reduced from LxLx21x21 to LxL matrix
# (by taking the L2norm for each of the 20x20). In the code below, you can access
# this as mtx["raw"]. Further correction (average product correction) is then performed
# to the mtx["raw"] to remove the effects of entropy, mtx["apc"]. The relative
# ranking of mtx["apc"] is used to assess importance. When there are enough effective
# sequences (>1000), we find that the top 1.0L contacts are ~90% accurate! When the
# number of effective sequences is lower, NN can help clean noise and fill in missing
# contacts.

# Functions for extracting contacts from MRF
###################


def normalize(x):
    x = stats.boxcox(x - np.amin(x) + 1.0)[0]
    x_mean = np.mean(x)
    x_std = np.std(x)
    return((x - x_mean) / x_std)


def get_mtx(mrf):
    '''get mtx given mrf'''
    # l2norm of 20x20 matrices (note: we ignore gaps)
    raw = np.sqrt(np.sum(np.square(mrf["w"][:, :-1, :-1]), (1, 2)))
    raw_sq = squareform(raw)

    # apc (average product correction)
    ap_sq = np.sum(raw_sq, 0, keepdims=True) * np.sum(raw_sq, 1, keepdims=True) / np.sum(raw_sq)
    apc = squareform(raw_sq - ap_sq, checks=False)

    mtx = {"i": mrf["w_idx"][:, 0],
           "j": mrf["w_idx"][:, 1],
           "raw": raw,
           "apc": apc,
           "zscore": normalize(apc)}
    return mtx


def run_gremlin(input_file: str, output_file: Optional[h5py.File] = None, max_seq_len: int = 700):
    # ===============================================================================
    # PREP MSA
    # ===============================================================================
    names, seqs = parse_fasta(input_file, limit=1000, max_seq_len=700)

    try:
        msa = mk_msa(seqs)
    except TooFewValidMatchesException:
        try:
            names, seqs = parse_fasta(input_file)
            msa = mk_msa(seqs)
        except TooFewValidMatchesException:
            raise TooFewValidMatchesException(input_file)

    mrf = GREMLIN(msa)
    mtx = get_mtx(mrf)

    this_protein_id = os.path.basename(input_file).split('.')[0]

    if output_file is not None:
        protein_group = output_file.create_group(this_protein_id)
        for key in ['v', 'w', 'raw', 'apc', 'v_idx', 'w_idx']:
            if key in mrf:
                array = mrf[key]
            elif key in mtx:
                array = mtx[key]
            dtype = array.dtype
            if dtype in [np.float32, np.float64]:
                array = np.asarray(array, np.float32)
                dtype_str = 'f'
            elif dtype in [np.int32, np.int64]:
                array = np.asarray(array, np.int32)
                dtype_str = 'i'
            else:
                raise ValueError("Unknown dtype {}".format(dtype))

            protein_group.create_dataset(
                key, dtype=dtype_str, data=array, compression='gzip')
    else:
        return msa, mrf, mtx


if __name__ == '__main__':
    import argparse
    from glob import glob
    from tqdm import tqdm

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    parser = argparse.ArgumentParser(description='Runs Gremlin_TF to output mrf from fasta/a2m file')
    # parser.add_argument('input_file', type=str, help='input fasta file')
    parser.add_argument('output_file', type=str, help='output h5py file')

    args = parser.parse_args()

    files = glob('/big/davidchan/roshan/raw/**a2m.gz')
    with tqdm(total=len(files)) as progress_bar:
        for shard in range(len(files) // 1000):
            output_file = args.output_file.split('.')[0]
            curr_out_file = output_file + f'_{shard}.h5'
            if os.path.exists(curr_out_file):
                progress_bar.update(1000)
                continue

            this_shard_files = files[1000 * shard:1000 * (shard + 1)]
            with h5py.File(curr_out_file, "a") as outfile:
                for input_file in this_shard_files:
                    this_protein_id = os.path.basename(input_file).split('.')[0]
                    if this_protein_id in outfile:
                        progress_bar.update()
                        continue

                    with contextlib.suppress(SequenceLengthException):
                        run_gremlin(input_file, outfile)

                    progress_bar.update()
