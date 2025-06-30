import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.nn import rnn_cell
import tf_slim as slim
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import random

xavier = tf.glorot_uniform_initializer

############### Global Parameters ###############
train_path = './data/features/training/'
test_path = './data/features/testing/'
demo_path = './data/external/demo_model'
default_model_path = './data/external/demo_model'
save_path = './model/'
video_path = './data/raw/videos/testing/positive/'
train_num = 128  # Updated batch count
test_num = 46

# Training Parameters
learning_rate = 0.0001
n_epochs = 30
batch_size = 10
# Network Parameters
n_input = 4096
n_detection = 20
n_hidden = 512
n_img_hidden = 256
n_att_hidden = 256
n_classes = 2
n_frames = 100

# Enhancement Parameters
focal_gamma = 2.0
focal_alpha = 0.25
l2_weight = 0.001
num_heads = 4
##################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Accident Prediction Model')
    parser.add_argument('--mode', dest='mode', help='train or test', default='demo')
    parser.add_argument('--model', dest='model', default=default_model_path)
    parser.add_argument('--gpu', dest='gpu', default='0')
    return parser.parse_args()


def build_model():
    x = tf.placeholder("float", [None, n_frames, n_detection, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep = tf.placeholder("float", [None])
    global_step = tf.Variable(0, trainable=False)

    # ========== Enhanced Weight Initialization ==========
    weights = {
        'em_obj': tf.get_variable('em_obj', [n_input, n_att_hidden],
                          initializer=xavier()),
        'em_img': tf.get_variable('em_img', [n_input, n_img_hidden],
                          initializer=xavier()),
        'att_w' : tf.get_variable('att_w', [n_att_hidden, 1],
                          initializer=xavier()),
        'att_wa': tf.get_variable('att_wa', [n_hidden, n_att_hidden],
                                initializer=xavier()),
        'att_ua': tf.get_variable('att_ua', [n_att_hidden, n_att_hidden],
                                initializer=xavier()),
        'out'   : tf.get_variable('out', [n_hidden, n_classes],
                          initializer=xavier())
    }
    biases = {
        'em_obj': tf.Variable(tf.zeros([n_att_hidden])),
        'em_img': tf.Variable(tf.zeros([n_img_hidden])),
        'att_ba': tf.Variable(tf.zeros([n_att_hidden])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # ========== Multi-Head Attention Implementation ==========
    def multi_head_attention(query, keys, values, num_heads=4):
        """ Multi-head scaled dot-product attention """
        head_dim = n_att_hidden // num_heads
        heads = []
        for h in range(num_heads):
            with tf.variable_scope(f'head_{h}'):
                # Linear projections
                Q = tf.layers.dense(query, head_dim, use_bias=False, name='Q_proj')
                K = tf.layers.dense(keys, head_dim, use_bias=False, name='K_proj')
                V = tf.layers.dense(values, head_dim, use_bias=False, name='V_proj')
                
                # Scaled dot-product attention
                d_k = tf.cast(tf.shape(K)[-1], tf.float32)
                scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
                weights = tf.nn.softmax(scores, axis=-1)
                head = tf.matmul(weights, V)
                heads.append(head)
                
        # Concatenate heads and apply final linear layer
        concat = tf.concat(heads, axis=-1)
        output = tf.layers.dense(concat, n_att_hidden, name='att_out')
        return output

    # ========== Positional Encoding ==========
    def positional_encoding(time_step, batch_sz, hidden_dim):
        pos  = tf.cast(time_step, tf.float32)              # scalar
        i    = tf.cast(tf.range(hidden_dim), tf.float32)   # [D]
        denom= tf.pow(10000.0, (i // 2) * 2.0 / hidden_dim)

        angle = pos / denom                                # [D]
        sin   = tf.sin(angle)
        cos   = tf.cos(angle)
        # interleave sin(0), cos(0), sin(1), cos(1) ...
        pe_row = tf.where(tf.equal(tf.mod(i, 2), 0), sin, cos)  # [D]

        pe_row = tf.reshape(pe_row, [1, hidden_dim])       # [1, D]
        return tf.tile(pe_row, [batch_sz, 1])              # [B, D]

    # LSTM cell with dropout
    lstm_cell = rnn_cell.LSTMCell(n_hidden,
                                initializer=xavier(),
                                use_peepholes=True,
                                state_is_tuple=False)
    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell, output_keep_prob=1 - keep[0]
    )
    
    istate = tf.zeros([batch_size, lstm_cell.state_size])
    h_prev = tf.zeros([batch_size, n_hidden])
    loss = 0.0
    zeros_object = tf.to_float(tf.not_equal(
        tf.reduce_sum(tf.transpose(x[:, :, 1:n_detection, :], [1, 2, 0, 3]), 3), 0))
    
    # ========== Main Temporal Processing Loop ==========
    for i in range(n_frames):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # Apply positional encoding
            time_enc = positional_encoding(i, batch_size, n_att_hidden)
            
            X = tf.transpose(x[:, i, :, :], [1, 0, 2])
            # Image feature embedding with positional encoding
            image = tf.matmul(X[0], weights['em_img']) + biases['em_img'] + time_enc
            
            # Object feature embedding with positional encoding
            n_object = tf.reshape(X[1:n_detection], [-1, n_input])
            n_object = tf.matmul(n_object, weights['em_obj']) + biases['em_obj']
            n_object = tf.reshape(n_object, [n_detection-1, batch_size, n_att_hidden])
            n_object = n_object + tf.expand_dims(time_enc, 0)  # Add time encoding
            n_object = n_object * tf.expand_dims(zeros_object[i], 2)
            
            # ========== Enhanced Attention Mechanism ==========
            image_part = tf.matmul(
                n_object,
                tf.tile(tf.expand_dims(weights['att_ua'], 0), [n_detection-1, 1, 1])
            ) + biases['att_ba']
            
            # Multi-head attention
            raw_att = multi_head_attention(
                query=h_prev,
                keys=tf.transpose(image_part, [1, 0, 2]),   # [B, Nobj, 256]
                values=tf.transpose(n_object , [1, 0, 2]),  # [B, Nobj, 256]
                num_heads=num_heads
            )
            # pool over objects → [B, 256]
            attention = tf.reduce_mean(raw_att, axis=1)      # or tf.reduce_max(…,1)

            fusion = tf.concat([image, attention], axis=1)   # shapes now both [B,256]
            
            # LSTM processing
            with tf.variable_scope('LSTM') as vs:
                outputs, istate = lstm_cell_dropout(fusion, istate)
                lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
            
            h_prev = outputs
            pred = tf.matmul(outputs, weights['out']) + biases['out']
            
            # Store predictions
            if i == 0:
                soft_pred = tf.reshape(                       # ←  ❌ fixed
                    tf.gather(tf.transpose(tf.nn.softmax(pred), (1, 0)), 1),
                    (batch_size, 1)
                )
                all_alphas = tf.expand_dims(tf.reduce_mean(attention, axis=-1), 0)
            else:
                soft_pred = tf.concat([
                    soft_pred,
                    tf.reshape(                               # ←  ❌ fixed
                        tf.gather(tf.transpose(tf.nn.softmax(pred), (1, 0)), 1),
                        (batch_size, 1)
                    )
                ], 1)
                all_alphas = tf.concat(
                    [all_alphas, tf.expand_dims(tf.reduce_mean(attention, axis=-1), 0)],
                    0
                )
            # ========== Enhanced Focal Loss ==========
            # Get predicted probabilities
            pred_probs = tf.nn.softmax(pred)
            
            # Focal loss calculation
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)
            
            # Class-balanced focal loss
            pt = tf.where(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), 
                         pred_probs[:, 1], 
                         1 - pred_probs[:, 1])
            
            focal_weight = focal_alpha * tf.pow(1.0 - pt, focal_gamma)
            focal_loss = focal_weight * cross_entropy
            
            # Temporal weighting (stronger for frames closer to accident)
            temporal_weight = tf.exp(-(n_frames - i - 1) / 15.0)
            temp_loss = tf.reduce_mean(temporal_weight * focal_loss)
            
            loss += temp_loss
    
    # ========== Regularization ==========
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                       if 'bias' not in v.name]) * l2_weight
    total_loss = loss / n_frames + l2_loss
    
    # ========== Optimizer with Learning Rate Decay ==========
    lr = tf.train.exponential_decay(
        learning_rate, global_step,
        decay_steps=train_num * 5,  # Decay every 5 epochs
        decay_rate=0.9, staircase=True
    )
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads, vars = zip(*optimizer.compute_gradients(total_loss))
    grads, _ = tf.clip_by_global_norm(grads, 5.0)  # Gradient clipping
    train_op = optimizer.apply_gradients(zip(grads, vars), global_step=global_step)
    
    return x, keep, y, train_op, total_loss, lstm_variables, soft_pred, all_alphas


# ========== Enhanced Data Augmentation ==========
def augment_batch(batch_xs, batch_ys):
    """Apply temporal and spatial augmentations"""
    # Temporal jitter (random shift)
    if random.random() < 0.3:  # 30% chance
        shift = random.randint(-5, 5)
        batch_xs = np.roll(batch_xs, shift, axis=1)
    
    # Feature noise
    if random.random() < 0.5:  # 50% chance
        noise = np.random.normal(0, 0.05, batch_xs.shape)
        batch_xs = batch_xs + noise
    
    # Class balancing (oversample positive examples)
    pos_idx = np.where(batch_ys[:, 1] == 1)[0]
    if len(pos_idx) < batch_size // 3 and len(pos_idx) > 0:
        dup = np.random.choice(pos_idx,
                               size=batch_size//3 - len(pos_idx),
                               replace=True)
        batch_xs = np.concatenate([batch_xs, batch_xs[dup]], axis=0)
        batch_ys = np.concatenate([batch_ys, batch_ys[dup]], axis=0)

    # ---------- NEW  ➜  trim / pad to fixed size -------------
    if batch_xs.shape[0] > batch_size:          # too many → random pick
        sel = np.random.choice(batch_xs.shape[0], batch_size, replace=False)
        batch_xs, batch_ys = batch_xs[sel], batch_ys[sel]
    elif batch_xs.shape[0] < batch_size:        # too few → pad duplicates
        pad = np.random.choice(batch_xs.shape[0], batch_size - batch_xs.shape[0],
                               replace=True)
        batch_xs = np.concatenate([batch_xs, batch_xs[pad]], axis=0)
        batch_ys = np.concatenate([batch_ys, batch_ys[pad]], axis=0)

    return batch_xs, batch_ys


def train(resume_from=None):
    x, keep, y, train_op, total_loss, lstm_variables, soft_pred, all_alpha = build_model()

    sess = tf.InteractiveSession(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)

    # ── 1. try to resume ─────────────────────────────────────────────
    start_epoch = 0
    if resume_from is None:                   # CLI arg or hard-coded path
        resume_from = tf.train.latest_checkpoint(save_path)
    if resume_from:
        saver.restore(sess, resume_from)
        print("✓ resumed from", resume_from)

        # the step number is appended to the ckpt name: model_epoch12-20000
        # -> grab the final “12”
        ckpt_name = os.path.basename(resume_from)
        parts = ckpt_name.split('-')[0].split('epoch')
        if len(parts) == 2 and parts[1].isdigit():
            start_epoch = int(parts[1])
        else:
            # or track it with a tf.Variable called global_step; then:
            # start_epoch = sess.run(global_step) // steps_per_epoch
            pass
    # ────────────────────────────────────────────────────────────────
    
    # Training loop
    for epoch in range(start_epoch, n_epochs):
        print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
        epoch_loss = []
        batch_indices = list(range(1, train_num+1))
        random.shuffle(batch_indices)
        t0 = time.time()
        
        for batch in batch_indices:
            # Load and augment data
            data = np.load(f"{train_path}batch_{batch:03d}.npz")
            batch_xs, batch_ys = augment_batch(data['data'], data['labels'])
            
            # Train step
            _, batch_loss = sess.run(
                    [train_op, total_loss],                 # ← use train_op here
                    feed_dict={x: batch_xs, y: batch_ys, keep: [0.5]}
            )
            epoch_loss.append(batch_loss)
            print(f"  Batch {batch}/{train_num} \t Loss: {batch_loss:.4f}")
        
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} \t Time: {time.time()-t0:.2f}s")
        
        # Save checkpoint and evaluate
        if (epoch+1) % 3 == 0 or epoch == n_epochs - 1:
            saver.save(sess, save_path + f"model_epoch{epoch+1}", global_step=epoch+1)
            print("Checkpoint saved. Running evaluation...")
            test_all(sess, train_num, train_path, x, keep, y, total_loss, lstm_variables, soft_pred, "Training")
            test_all(sess, test_num, test_path, x, keep, y, total_loss, lstm_variables, soft_pred, "Testing")

    print("Training finished. Saving final model...")
    saver.save(sess, save_path + "final_model")


def test_all(sess, num, path, x, keep, y, loss, lstm_variables, soft_pred, dataset_name):
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for num_batch in range(1, num+1):
        # Load test data
        file_name = f'{num_batch:03d}'
        test_data = np.load(path + 'batch_' + file_name + '.npz')
        test_x = test_data['data']
        test_y = test_data['labels']
        
        # Run inference
        batch_loss, pred = sess.run(
            [loss, soft_pred],
            feed_dict={x: test_x, y: test_y, keep: [0.0]}
        )
        total_loss += batch_loss
        
        # Store results
        all_preds.append(pred[:, 0:90])
        all_labels.append(test_y[:, 1].reshape(-1, 1))
    
    # Combine results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    avg_loss = total_loss / num
    auc, ap = calculate_auc(all_preds, all_labels)
    
    print(f"\n{dataset_name} Set Evaluation:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  AP: {ap:.4f}")
    
    # Precision@K metrics
    for k in [10, 30, 50]:
        topk_preds = np.sort(all_preds, axis=1)[:, -k:]
        precision = np.mean([np.max(topk_preds[i]) > 0.5 for i in range(len(all_preds))])
        print(f"  Precision@{k}: {precision:.4f}")


def calculate_auc(preds, labels):
    """Calculate AUC and Average Precision"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Flatten predictions and labels
    flat_preds = preds.flatten()
    flat_labels = np.tile(labels, (1, preds.shape[1])).flatten()
    
    # Calculate metrics
    auc = roc_auc_score(flat_labels, flat_preds)
    ap = average_precision_score(flat_labels, flat_preds)
    return auc, ap


# The rest of the functions (evaluation, vis, test) remain similar but 
# will need adjustments to handle the enhanced model outputs.

# Due to character limits, I've focused on the core enhancements.
# Let me know if you need the complete evaluation function updated.

def evaluation(all_pred, all_labels, total_time=90, vis=False, length=None):
    """
    Enhanced evaluation with comprehensive metrics:
    - Precision-Recall curve (AUC)
    - Time to Accident analysis
    - Precision@K metrics
    - Confusion matrix statistics
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score, 
                               precision_recall_curve, confusion_matrix)
    
    # 1. Basic setup
    if length is None:
        length = [total_time] * all_pred.shape[0]
    
    # 2. Calculate core metrics
    flat_pred = all_pred.flatten()
    flat_labels = np.tile(all_labels, (1, all_pred.shape[1])).flatten()
    
    # AUC and AP
    auc = roc_auc_score(flat_labels, flat_pred)
    ap = average_precision_score(flat_labels, flat_pred)
    
    # Precision@K
    precision_at_k = {}
    for k in [10, 30, 50, 70]:
        topk = np.sort(all_pred, axis=1)[:, -k:]
        precision_at_k[k] = np.mean(topk.max(axis=1) > 0.5)
    
    # 3. Time to Accident analysis
    warning_times = []
    for i in range(len(all_labels)):
        if all_labels[i, 0] == 1:  # Accident video
            accident_frame = length[i] - 10  # Last 10 frames contain accident
            risk_frames = np.where(all_pred[i] > 0.7)[0]
            
            if len(risk_frames) > 0:
                first_warning = risk_frames[0]
                warning_time = (accident_frame - first_warning) / 20.0  # FPS=20
                warning_times.append(warning_time)
    
    mean_warning_time = np.mean(warning_times) if warning_times else 0
    
    # 4. Confusion matrix at optimal threshold
    precision, recall, thresholds = precision_recall_curve(flat_labels, flat_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (flat_pred > optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(flat_labels, y_pred).ravel()
    
    # 5. Print comprehensive report
    print("\n=== Enhanced Evaluation Metrics ===")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"Mean Warning Time: {mean_warning_time:.2f}s")
    print(f"Optimal Threshold: {optimal_threshold:.4f} (F1={f1_scores[optimal_idx]:.4f})")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("Precision@K:")
    for k, p in precision_at_k.items():
        print(f"  @{k}: {p:.4f}")
    
    # 6. Visualization
    if vis:
        plt.figure(figsize=(15, 10))
        
        # Precision-Recall Curve
        plt.subplot(2, 2, 1)
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # Time to Accident Histogram
        plt.subplot(2, 2, 2)
        plt.hist(warning_times, bins=20, alpha=0.7)
        plt.axvline(mean_warning_time, color='r', linestyle='dashed')
        plt.xlabel('Warning Time (seconds)')
        plt.ylabel('Count')
        plt.title(f'Time to Accident (Mean: {mean_warning_time:.2f}s)')
        
        # Risk Profile Examples
        plt.subplot(2, 2, 3)
        for i in random.sample(range(len(all_pred)), min(5, len(all_pred))):
            plt.plot(all_pred[i], alpha=0.7, label=f'Video {i+1}')
        plt.axhline(optimal_threshold, color='r', linestyle='--')
        plt.ylim(0, 1)
        plt.xlabel('Frame')
        plt.ylabel('Risk Score')
        plt.title('Sample Risk Profiles')
        plt.legend()
        
        # Metric Summary
        plt.subplot(2, 2, 4)
        metrics = ['AUC', 'AP', 'Warning Time']
        values = [auc, ap, mean_warning_time]
        plt.bar(metrics, values, color=['blue', 'green', 'orange'])
        plt.ylabel('Value')
        plt.title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('./evaluation_report.png')
        plt.show()
    
    return {
        'auc': auc,
        'ap': ap,
        'warning_time': mean_warning_time,
        'precision_at_k': precision_at_k,
        'confusion_matrix': (tp, fp, tn, fn),
        'optimal_threshold': optimal_threshold
    }


def vis(model_path):
    """Enhanced visualization with multi-head attention support"""
    # Build model
    x, keep, y, _, _, _, soft_pred, all_alphas = build_model()
    
    # Initialize session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.InteractiveSession(config=tf.ConfigProto(
        allow_soft_placement=True, gpu_options=gpu_options
    ))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    # Create output directory
    os.makedirs('./visualizations', exist_ok=True)
    
    # Process each test batch
    for num_batch in range(1, test_num):
        file_name = f'{num_batch:03d}'
        data = np.load(os.path.join(demo_path, f'batch_{file_name}.npz'))
        video_data = data['data']
        video_labels = data['labels']
        video_det = data['det']
        video_ids = data['ID']
        
        # Run inference
        preds, attentions = sess.run(
            [soft_pred, all_alphas],
            feed_dict={x: video_data, y: video_labels, keep: [0.0]}
        )
        
        # Process each video in batch
        for i in range(len(video_ids)):
            vid = video_ids[i].decode('utf-8') if isinstance(video_ids[i], bytes) else video_ids[i]
            if video_labels[i][1] == 1:  # Only visualize accident videos
                print(f"\nVisualizing: {vid}")
                
                # 1. Risk curve plot
                plt.figure(figsize=(12, 4))
                plt.plot(preds[i], 'r-', linewidth=2)
                plt.axhline(y=0.7, color='b', linestyle='--')
                plt.title(f'Risk Prediction: {vid}')
                plt.xlabel('Frame')
                plt.ylabel('Accident Probability')
                plt.ylim(0, 1)
                plt.savefig(f'./visualizations/{vid}_risk_curve.png')
                
                # 2. Attention heatmap (across heads)
                plt.figure(figsize=(10, 6))
                head_attention = attentions[:, :, i]  # [frames, objects]
                plt.imshow(head_attention.T, aspect='auto', cmap='hot')
                plt.colorbar()
                plt.title(f'Attention Map: {vid}')
                plt.xlabel('Frame')
                plt.ylabel('Object Index')
                plt.savefig(f'./visualizations/{vid}_attention_heatmap.png')
                
                # 3. Video annotation
                cap = cv2.VideoCapture(os.path.join(video_path, f'{vid}.mp4'))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Prepare video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    f'./visualizations/{vid}_annotated.mp4',
                    fourcc, fps, (width, height)
                )

                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_idx >= n_frames:
                        break
                    
                    # Draw bounding boxes with attention
                    for obj_idx in range(n_detection - 1):
                        attn = attentions[frame_idx, obj_idx, i]
                        bbox = video_det[i, frame_idx, obj_idx + 1]
                        x0, y0, x1, y1 = map(int, bbox[:4])
                        
                        # Color based on attention strength
                        color = (0, int(255 * min(attn, 1.0)), 0)
                        thickness = 2 if attn > 0.3 else 1
                        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
                        
                        # Label with attention score
                        cv2.putText(frame, f'{attn:.2f}', (x0, y0-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Add risk score overlay
                    risk = preds[i, frame_idx]
                    color = (0, 0, 255) if risk > 0.7 else (0, 255, 0)
                    cv2.putText(frame, f'Risk: {risk:.2f}', (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Add frame marker if accident is near
                    if frame_idx > n_frames - 10:
                        cv2.putText(frame, 'ACCIDENT IMMINENT', (width//2-150, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    out.write(frame)
                    frame_idx += 1
                
                cap.release()
                out.release()
                print(f"Saved visualization for {vid}")
                
                plt.close('all')  # Free memory


def test(model_path):
    """Testing function updated with new metrics"""
    x, keep, y, optimizer, loss, lstm_variables, soft_pred, all_alphas = build_model()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.InteractiveSession(config=tf.ConfigProto(
        allow_soft_placement=True, gpu_options=gpu_options
    ))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    print("Model restored!")
    print("Training Set Evaluation:")
    test_all(sess, train_num, train_path, x, keep, y, loss, lstm_variables, soft_pred, "Training")
    print("\nTesting Set Evaluation:")
    test_all(sess, test_num, test_path, x, keep, y, loss, lstm_variables, soft_pred, "Testing")


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu else '0'
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test(args.model)
    elif args.mode == 'demo':
        # Update to use enhanced visualization
        vis(args.model)