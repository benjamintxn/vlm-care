#!/usr/bin/env python3
# train_from_batches.py
#
# Train the Dynamic-Spatial-Attention RNN directly from pre-batched .npz files,
# with dynamic batch sizes, early stopping, L2 regularisation, and dropout.

import os, glob, time, random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.nn import rnn_cell

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FEATURE_BATCH_DIR  = './data/annotated_batches/training'   # root with /positive & /negative
SAVE_PREFIX        = './model/phase1_'                   # checkpoint prefix
DEFAULT_BATCH_SIZE = 10                                  # for logging

N_FRAMES           = 100
N_DET              = 20
N_INPUT            = 4096                                # appearance+flow
N_HIDDEN           = 512
N_IMG_HIDDEN       = 256
N_ATT_HIDDEN       = 256
LEARNING_RATE      = 1e-4
N_EPOCHS           = 30
LOSS_WEIGHT_ATTN   = 1e-2
L2_REG             = 1e-5                                # L2 weight decay
EARLY_STOP_PATIENCE= 5                                   # epochs with no val improvement
VAL_BATCHES        = 20                                  # number of batches to hold out
# ──────────────────────────────────────────────────────────────────────────────

# ─── Data helpers ─────────────────────────────────────────────────────────────

def list_batch_files(root):
    pos = sorted(glob.glob(os.path.join(root, 'positive', 'batch_*.npz')))
    neg = sorted(glob.glob(os.path.join(root, 'negative', 'batch_*.npz')))
    return pos + neg


def load_npz_batch(path):
    arr    = np.load(path, allow_pickle=True)
    data   = arr['data'].astype(np.float32)
    labels = arr['labels'].astype(np.float32)
    if 'attn_sup' in arr:
        attn = arr['attn_sup'].astype(np.float32)
    else:
        B = data.shape[0]
        attn = np.zeros((B, N_FRAMES, N_DET), np.float32)
    return data, labels, attn
# ───────────────────────────────────────────────────────────────────────────────

# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    x_ph    = tf.placeholder(tf.float32, [None, N_FRAMES, N_DET, N_INPUT], name='x')
    y_ph    = tf.placeholder(tf.float32, [None, 2],               name='y')
    attn_ph = tf.placeholder(tf.float32, [None, N_FRAMES, N_DET], name='attn_sup')
    keep_ph = tf.placeholder(tf.float32, [],                      name='keep')

    batch_size = tf.shape(x_ph)[0]

    # weights & biases
    weights = {
        'em_obj': tf.Variable(tf.random_normal([N_INPUT, N_ATT_HIDDEN], stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([N_INPUT, N_IMG_HIDDEN], stddev=0.01)),
        'att_w' : tf.Variable(tf.random_normal([N_ATT_HIDDEN, 1], stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([N_HIDDEN, N_ATT_HIDDEN], stddev=0.01)),
        'att_ua': tf.Variable(tf.random_normal([N_ATT_HIDDEN, N_ATT_HIDDEN], stddev=0.01)),
        'out'   : tf.Variable(tf.random_normal([N_HIDDEN, 2], stddev=0.01))
    }
    biases = {
        'em_obj': tf.Variable(tf.random_normal([N_ATT_HIDDEN], stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([N_IMG_HIDDEN], stddev=0.01)),
        'att_ba': tf.Variable(tf.zeros([N_ATT_HIDDEN])),
        'out'   : tf.Variable(tf.random_normal([2], stddev=0.01))
    }

    lstm = rnn_cell.LSTMCell(N_HIDDEN, state_is_tuple=False)
    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1.0 - keep_ph)
    istate = tf.zeros([batch_size, lstm.state_size])
    hprev  = tf.zeros([batch_size, N_HIDDEN])
    loss   = 0.0

    zeros_obj = tf.cast(
        tf.not_equal(
            tf.reduce_sum(
                tf.transpose(x_ph[:, :, 1:N_DET, :], [1, 2, 0, 3]), axis=3),
            0.0),
        tf.float32)

    all_alphas = []
    for t in range(N_FRAMES):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            Xi = tf.transpose(x_ph[:, t, :, :], [1, 0, 2])
            img_feat = tf.matmul(Xi[0], weights['em_img']) + biases['em_img']
            objs = tf.reshape(Xi[1:], [-1, N_INPUT])
            objs = tf.matmul(objs, weights['em_obj']) + biases['em_obj']
            objs = tf.reshape(objs, [N_DET - 1, batch_size, N_ATT_HIDDEN])
            objs *= tf.expand_dims(zeros_obj[t], 2)

            brcst_w = tf.tile(tf.expand_dims(weights['att_w'], 0), [N_DET - 1, 1, 1])
            img_part = tf.matmul(
                objs,
                tf.tile(tf.expand_dims(weights['att_ua'], 0), [N_DET - 1, 1, 1])
            ) + biases['att_ba']

            e = tf.tanh(tf.matmul(hprev, weights['att_wa']) + img_part)
            alphas = tf.nn.softmax(
                tf.reduce_sum(tf.matmul(e, brcst_w), 2), 0) * zeros_obj[t]
            all_alphas.append(tf.expand_dims(alphas, 0))
            attention = tf.reduce_sum(tf.expand_dims(alphas, 2) * objs, 0)

            fusion = tf.concat([img_feat, attention], 1)
            out, istate = lstm(fusion, istate)
            hprev = out
            logits = tf.matmul(out, weights['out']) + biases['out']

            wpos = tf.exp(-(N_FRAMES - t - 1) / 20.0)
            cel  = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_ph)
            posl = -wpos * cel * y_ph[:, 1]
            negl = cel * y_ph[:, 0]
            loss += tf.reduce_mean(posl + negl)

    all_alphas = tf.concat(all_alphas, 0)
    attn_pred = tf.transpose(all_alphas, [2, 0, 1])
    attn_loss = tf.reduce_mean(tf.square(attn_pred - attn_ph[:, :, 1:]))

    # L2 regularisation on all weight matrices
    l2_loss = tf.add_n([tf.nn.l2_loss(W) for W in weights.values()]) * L2_REG

    total_loss = loss / N_FRAMES + LOSS_WEIGHT_ATTN * attn_loss + l2_loss
    optimizer  = tf.train.AdamOptimizer(LEARNING_RATE)
    gvs        = optimizer.compute_gradients(total_loss)
    capped_gvs = [(tf.clip_by_norm(g, 5.0), v) for g, v in gvs]
    train_op   = optimizer.apply_gradients(capped_gvs)

    return x_ph, y_ph, attn_ph, keep_ph, train_op, total_loss
# ───────────────────────────────────────────────────────────────────────────────

# ─── Training ──────────────────────────────────────────────────────────────────

def train():
    all_files = list_batch_files(FEATURE_BATCH_DIR)
    if not all_files:
        raise RuntimeError('No batch files found!')

    # select random validation set
    val_files = random.sample(all_files, min(VAL_BATCHES, len(all_files)))
    train_files = [f for f in all_files if f not in val_files]
    print(f'Training on {len(train_files)} batches, validating on {len(val_files)} batches')

    x_ph, y_ph, attn_ph, keep_ph, train_op, loss_op = build_model()

    cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)

        best_val, patience = float('inf'), 0
        for epoch in range(1, N_EPOCHS+1):
            random.shuffle(train_files)
            t0 = time.time()

            # training
            for i, fn in enumerate(train_files, 1):
                bx, by, ba = load_npz_batch(fn)
                feed = {x_ph: bx, y_ph: by, attn_ph: ba, keep_ph: 0.5}
                _, tr_loss = sess.run([train_op, loss_op], feed_dict=feed)
                if i % 10 == 0:
                    print(f'Epoch {epoch} Batch {i}/{len(train_files)}  Loss={tr_loss:.4f}')

            # validation
            val_losses = []
            for fn in val_files:
                bx, by, ba = load_npz_batch(fn)
                l = sess.run(loss_op, feed_dict={x_ph:bx, y_ph:by, attn_ph:ba, keep_ph:1.0})
                val_losses.append(l)
            mean_val = np.mean(val_losses)
            print(f'Epoch {epoch} done in {time.time()-t0:.1f}s  Val_loss={mean_val:.4f}')

            # early stopping
            if mean_val < best_val:
                best_val, patience = mean_val, 0
                saver.save(sess, SAVE_PREFIX + '_best', global_step=epoch)
                print('  🎉 New best model saved')
            else:
                patience += 1
                print(f'  ⚠️  No improvement, patience {patience}/{EARLY_STOP_PATIENCE}')
                if patience >= EARLY_STOP_PATIENCE:
                    print('Early stopping — no val improvement')
                    break

if __name__ == '__main__':
    train()
