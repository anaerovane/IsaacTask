import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error 
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Layer, GlobalAveragePooling1D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import math

class DistanceWeightedSelfAttention(Layer):
    def __init__(self, d_model, lambda_decay=0.1, **kwargs):
        super(DistanceWeightedSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.lambda_decay = lambda_decay
        self.query_dense = Dense(d_model, use_bias=False, name='query_dense_attention')
        self.key_dense = Dense(d_model, use_bias=False, name='key_dense_attention')
        self.value_dense = Dense(d_model, use_bias=False, name='value_dense_attention')
        self.final_dense = Dense(d_model, use_bias=False, name='final_dense_attention')

    def build(self, input_shape):
        super(DistanceWeightedSelfAttention, self).build(input_shape)

    def call(self, inputs, allele_sizes, mask=None):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        attention_scores = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = attention_scores / tf.math.sqrt(dk)
        allele_sizes_float = tf.cast(allele_sizes, tf.float32)
        distance_matrix = tf.abs(tf.expand_dims(allele_sizes_float, axis=1) - tf.expand_dims(allele_sizes_float, axis=2))
        distance_decay = tf.exp(-self.lambda_decay * distance_matrix)
        scaled_attention_logits *= distance_decay
        if mask is not None:
            attention_mask = tf.expand_dims(mask, axis=1)
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -1e9
            scaled_attention_logits += adder
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        context_vector = tf.matmul(attention_weights, value)
        output = self.final_dense(context_vector)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "lambda_decay": self.lambda_decay})
        return config

class CastToFloat32Layer(Layer):
    def __init__(self, **kwargs):
        super(CastToFloat32Layer, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    def get_config(self):
        return super().get_config()

def parse_ratio_from_sample_file_name(sample_file_name_series):
    parsed_ratios = []
    pattern = re.compile(r'(?:[\d_]+)-(\d+(?:;\d+)*)-[A-Za-z]')
    for name in sample_file_name_series:
        match = pattern.search(name)
        if match: parsed_ratios.append(match.group(1))
        else: parsed_ratios.append(None)
    return parsed_ratios

def convert_ratio_to_proportions(ratio_str, num_contributors):
    if ratio_str is None: return None
    try:
        parts = [float(p) for p in ratio_str.split(';')]
        if len(parts) != num_contributors:
            if num_contributors == 0 and not parts: return []
            if num_contributors == 1 and len(parts) == 1: pass
            else: return None
        if num_contributors == 0: return []
        total = sum(parts)
        if total == 0: return [0.0] * num_contributors
        return [p / total for p in parts]
    except: return None

def load_and_extract_ratio_info(csv_file_path, max_people_for_ratio_task):
    try:
        df = pd.read_csv(csv_file_path)
        print(f"文件 '{csv_file_path}' 加载成功。行数: {len(df)}, 列数: {len(df.columns)}")
    except Exception as e:
        print(f"加载文件 '{csv_file_path}' 时发生错误: {e}")
        return None, None
    required_columns_for_ratio = ['Sample File', 'Predicted Number of People']
    for col in required_columns_for_ratio:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少用于解析N和比例的必需列 '{col}'。")
            return None, None
    sample_info_df = df[['Sample File', 'Predicted Number of People']].drop_duplicates().reset_index(drop=True)
    sample_info_df['N_Contributors'] = sample_info_df['Predicted Number of People'].astype(int)
    sample_info_df['Raw_Ratio_Str'] = parse_ratio_from_sample_file_name(sample_info_df['Sample File'])
    proportions_list, valid_samples_count, parsed_successfully_but_n_mismatch = [], 0, 0
    for index, row in sample_info_df.iterrows():
        n_contrib, raw_ratio = row['N_Contributors'], row['Raw_Ratio_Str']
        proportions = convert_ratio_to_proportions(raw_ratio, n_contrib) if raw_ratio is not None else None
        if proportions is not None:
            if (n_contrib == 0 and not proportions) or (n_contrib > 0 and len(proportions) == n_contrib):
                padded_proportions = proportions + [0.0] * (max_people_for_ratio_task - len(proportions))
                if len(padded_proportions) <= max_people_for_ratio_task or n_contrib == 0 :
                    proportions_list.append(padded_proportions)
                    valid_samples_count +=1
                else: proportions_list.append([None]*max_people_for_ratio_task) 
            else:
                proportions_list.append([None] * max_people_for_ratio_task)
                if raw_ratio is not None: parsed_successfully_but_n_mismatch +=1
        else: proportions_list.append([None] * max_people_for_ratio_task)
    sample_info_df['Target_Proportions_Padded'] = proportions_list
    y_target_for_loss_list = []
    for index, row in sample_info_df.iterrows():
        padded_props, n_val = row['Target_Proportions_Padded'], row['N_Contributors']
        if any(p is None for p in padded_props): y_target_for_loss_list.append([None] * (max_people_for_ratio_task + 1))
        else: y_target_for_loss_list.append(padded_props + [float(n_val)])
    sample_info_df['Y_Target_For_Loss'] = y_target_for_loss_list
    print(len(sample_info_df))
    print(valid_samples_count)
    if parsed_successfully_but_n_mismatch > 0: print(f" {parsed_successfully_but_n_mismatch} 个样本原始比例解析成功，但与N值不符或总和为0。")
    sample_info_df_filtered = sample_info_df[sample_info_df['Y_Target_For_Loss'].apply(lambda x: isinstance(x, list) and not any(val is None for val in x))].copy()
    print(f"过滤后有效样本数量 (N和比例解析): {len(sample_info_df_filtered)}")
    return df, sample_info_df_filtered

def preprocess_single_marker_inputs(
    original_wide_df, 
    sample_info_df_with_targets, 
    target_marker_name, 
    max_alleles_per_marker, 
    max_sequence_length, 
    num_features_per_allele_event=1
    ):
    sample_files_for_model = sample_info_df_with_targets['Sample File'].tolist()
    X_marker_features_single, X_marker_allele_sizes_single, X_marker_masks_single = [], [], []
    X_N_values_list, Y_targets_list = [], []
    all_sizes_marker_specific, all_heights_marker_specific = [], []

    if 'Marker' not in original_wide_df.columns:
        return None, None, None, None, None
    marker_subset_df = original_wide_df[original_wide_df['Marker'] == target_marker_name]
    
    for i in range(1, max_alleles_per_marker + 1):
        size_col, height_col = None, None
        if f'Size {i}' in marker_subset_df.columns: size_col = f'Size {i}'
        elif f'Size{i}' in marker_subset_df.columns: size_col = f'Size{i}'
        if f'Height {i}' in marker_subset_df.columns: height_col = f'Height {i}'
        elif f'Height{i}' in marker_subset_df.columns: height_col = f'Height{i}'
        if size_col: all_sizes_marker_specific.extend(pd.to_numeric(marker_subset_df[size_col], errors='coerce').dropna().tolist())
        if height_col: all_heights_marker_specific.extend(pd.to_numeric(marker_subset_df[height_col], errors='coerce').dropna().tolist())

    size_scaler, height_scaler = MinMaxScaler(), MinMaxScaler()
    if all_sizes_marker_specific: size_scaler.fit(np.array(all_sizes_marker_specific).reshape(-1, 1))
    if all_heights_marker_specific: height_scaler.fit(np.array(all_heights_marker_specific).reshape(-1, 1))

    for sample_idx, sample_file_name in enumerate(sample_files_for_model):
        sample_target_info = sample_info_df_with_targets[sample_info_df_with_targets['Sample File'] == sample_file_name].iloc[0]
        X_N_values_list.append([sample_target_info['N_Contributors']])
        Y_targets_list.append(sample_target_info['Y_Target_For_Loss'])
        marker_row_df = original_wide_df[(original_wide_df['Sample File'] == sample_file_name) & (original_wide_df['Marker'] == target_marker_name)]
        if marker_row_df.empty:
            X_marker_features_single.append(np.zeros((max_sequence_length, num_features_per_allele_event)))
            X_marker_allele_sizes_single.append(np.zeros(max_sequence_length))
            X_marker_masks_single.append(np.zeros(max_sequence_length, dtype=bool))
            continue
        current_marker_data_series = marker_row_df.iloc[0]
        allele_size_height_pairs = []
        for i in range(1, max_alleles_per_marker + 1):
            size_col, height_col = None, None
            if f'Size {i}' in current_marker_data_series.index: size_col = f'Size {i}'
            elif f'Size{i}' in current_marker_data_series.index: size_col = f'Size{i}'
            if f'Height {i}' in current_marker_data_series.index: height_col = f'Height {i}'
            elif f'Height{i}' in current_marker_data_series.index: height_col = f'Height{i}'
            if size_col and height_col:
                size_val = pd.to_numeric(current_marker_data_series.get(size_col), errors='coerce')
                height_val = pd.to_numeric(current_marker_data_series.get(height_col), errors='coerce')
                if pd.notna(size_val) and pd.notna(height_val) and height_val > 0:
                    allele_size_height_pairs.append({'Size': float(size_val), 'Height': float(height_val)})
        allele_size_height_pairs.sort(key=lambda x: x['Size'])
        sizes_raw = np.array([p['Size'] for p in allele_size_height_pairs])
        heights_raw = np.array([p['Height'] for p in allele_size_height_pairs])
        sizes_norm, heights_norm = np.zeros_like(sizes_raw, dtype=float), np.zeros_like(heights_raw, dtype=float)
        if len(sizes_raw) > 0 and all_sizes_marker_specific and hasattr(size_scaler, 'n_samples_seen_') and size_scaler.n_samples_seen_ > 0 :
            sizes_norm = size_scaler.transform(sizes_raw.reshape(-1,1)).flatten()
        elif len(sizes_raw) > 0: temp_s_scaler = MinMaxScaler(); sizes_norm = temp_s_scaler.fit_transform(sizes_raw.reshape(-1,1)).flatten()
        if len(heights_raw) > 0 and all_heights_marker_specific and hasattr(height_scaler, 'n_samples_seen_') and height_scaler.n_samples_seen_ > 0:
            heights_norm = height_scaler.transform(heights_raw.reshape(-1,1)).flatten()
        elif len(heights_raw) > 0: temp_h_scaler = MinMaxScaler(); heights_norm = temp_h_scaler.fit_transform(heights_raw.reshape(-1,1)).flatten()
        features_attention, sizes_attention = heights_norm, sizes_norm       
        seq_len = len(features_attention)
        padded_features = np.zeros((max_sequence_length, num_features_per_allele_event))
        if seq_len > 0: padded_features[:min(seq_len, max_sequence_length), 0] = features_attention[:min(seq_len, max_sequence_length)]
        padded_sizes = np.zeros(max_sequence_length)
        if seq_len > 0: padded_sizes[:min(seq_len, max_sequence_length)] = sizes_attention[:min(seq_len, max_sequence_length)]
        mask = np.zeros(max_sequence_length, dtype=bool)
        if seq_len > 0: mask[:min(seq_len, max_sequence_length)] = True
        X_marker_features_single.append(padded_features); X_marker_allele_sizes_single.append(padded_sizes); X_marker_masks_single.append(mask)
    return (np.array(X_marker_features_single), np.array(X_marker_allele_sizes_single), 
            np.array(X_marker_masks_single), np.array(X_N_values_list), np.array(Y_targets_list))

def build_single_marker_ratio_model(
    max_sequence_length, num_features_per_allele_event, 
    max_people_for_ratio_task, attention_d_model=128, lambda_decay_attention=0.01, 
    dense_units=256, dropout_rate=0.0, marker_name_prefix="marker" 
):
    marker_feature_input = Input(shape=(max_sequence_length, num_features_per_allele_event), name=f'{marker_name_prefix}_features_input')
    marker_allele_size_input = Input(shape=(max_sequence_length,), name=f'{marker_name_prefix}_allele_sizes_input')
    marker_mask_input = Input(shape=(max_sequence_length,), dtype=tf.bool, name=f'{marker_name_prefix}_mask_input')
    num_contributors_input = Input(shape=(1,), name='num_contributors_input', dtype='int32')

    attention_layer = DistanceWeightedSelfAttention(d_model=attention_d_model, lambda_decay=lambda_decay_attention, name=f'{marker_name_prefix}_attention')
    attention_output_sequence = attention_layer(marker_feature_input, marker_allele_size_input, mask=marker_mask_input)
    pooled_attention_output = GlobalAveragePooling1D(name=f'{marker_name_prefix}_gap')(attention_output_sequence, mask=marker_mask_input)
    n_input_float = CastToFloat32Layer(name='cast_N_to_float')(num_contributors_input) 
    combined_features = Concatenate(name=f'{marker_name_prefix}_combine_with_N')([pooled_attention_output, n_input_float])

    dense_layer = Dense(dense_units, activation='relu', name=f'{marker_name_prefix}_dense_1')(combined_features)
    if dropout_rate > 0: dense_layer = Dropout(dropout_rate, name=f'{marker_name_prefix}_dropout_1')(dense_layer)
    dense_layer = Dense(dense_units // 2, activation='relu', name=f'{marker_name_prefix}_dense_2')(dense_layer)
    if dropout_rate > 0: dense_layer = Dropout(dropout_rate, name=f'{marker_name_prefix}_dropout_2')(dense_layer)
    dense_layer = Dense(dense_units // 4, activation='relu', name=f'{marker_name_prefix}_dense_3')(dense_layer) 

    ratio_predictions_raw = Dense(max_people_for_ratio_task, activation='linear', name=f'{marker_name_prefix}_ratio_predictions_raw')(dense_layer)
    model_inputs = [marker_feature_input, marker_allele_size_input, marker_mask_input, num_contributors_input]
    model = Model(inputs=model_inputs, outputs=ratio_predictions_raw)
    return model

def custom_jsd_loss(y_true_padded_with_N, y_pred_raw_logits):
    num_contributors_true_batch = tf.cast(y_true_padded_with_N[:, -1], dtype=tf.int32)
    y_true_proportions_padded = y_true_padded_with_N[:, :-1]
    epsilon = tf.keras.backend.epsilon() 

    def calculate_jsd_for_sample(args):
        true_props_padded_single, pred_logits_single, n_single = args
        
        def jsd_fn():
            P_orig = true_props_padded_single[:n_single]       
            raw_Q_logits = pred_logits_single[:n_single]            
            Q_orig = tf.nn.softmax(raw_Q_logits) 
            P_safe = P_orig + epsilon
            P_safe = P_safe / tf.reduce_sum(P_safe, axis=-1, keepdims=True)
            Q_safe = Q_orig + epsilon
            Q_safe = Q_safe / tf.reduce_sum(Q_safe, axis=-1, keepdims=True)
            M = 0.5 * (P_safe + Q_safe)
            M_safe = M / tf.reduce_sum(M, axis=-1, keepdims=True) 
            M_safe = M_safe + epsilon
            M_safe = M_safe / tf.reduce_sum(M_safe, axis=-1, keepdims=True)

            kl_p_m = tf.keras.losses.kullback_leibler_divergence(P_safe, M_safe)
            kl_q_m = tf.keras.losses.kullback_leibler_divergence(Q_safe, M_safe)
            
            jsd = 0.5 * (kl_p_m + kl_q_m)
            return jsd

        return tf.cond(tf.greater(n_single, 0), 
                       jsd_fn, 
                       lambda: tf.constant(0.0, dtype=tf.float32))

    losses_per_sample = tf.map_fn(
        calculate_jsd_for_sample,
        (y_true_proportions_padded, y_pred_raw_logits, num_contributors_true_batch),
        fn_output_signature=tf.float32
    )
    
    valid_n_mask_float = tf.cast(tf.greater(num_contributors_true_batch, 0), tf.float32)
    num_valid_samples = tf.reduce_sum(valid_n_mask_float)
    total_loss = tf.reduce_sum(losses_per_sample) 
    
    final_loss = tf.cond(tf.greater(num_valid_samples, 0),
                       lambda: total_loss / num_valid_samples,
                       lambda: tf.constant(0.0, dtype=tf.float32)) 
    return final_loss

def get_predictions_and_compare(model, X_inputs_list, Y_targets_all_samples, sample_files_all_samples, max_people_config, marker_name_for_output=""):
    predictions_raw = model.predict(X_inputs_list) 
    results = []
    overall_mae_list = []
    for i in range(len(sample_files_all_samples)):
        sample_file = sample_files_all_samples[i]
        true_y_with_n = Y_targets_all_samples[i]
        true_n = int(true_y_with_n[-1])
        true_proportions_padded = true_y_with_n[:-1]
        true_proportions_actual = np.array(true_proportions_padded[:true_n] if true_n > 0 else [], dtype=float)
        pred_logits_sample = predictions_raw[i]
        pred_proportions_actual_list, relevant_logits_np = [], np.array([])
        mae_sample = np.nan
        if true_n > 0:
            relevant_logits = pred_logits_sample[:true_n]
            relevant_logits_np = relevant_logits
            pred_proportions_actual_tf = tf.nn.softmax(relevant_logits)
            pred_proportions_actual_np = pred_proportions_actual_tf.numpy()
            pred_proportions_actual_list = pred_proportions_actual_np.tolist()
            if len(true_proportions_actual) == len(pred_proportions_actual_np):
                 mae_sample = mean_absolute_error(true_proportions_actual, pred_proportions_actual_np)
                 overall_mae_list.append(mae_sample)
        
        entry = {
            "Sample File": sample_file, 
            "N Contributors": true_n,
            f"True Proportions ({marker_name_for_output})" if marker_name_for_output else "True Proportions": [round(p, 4) for p in true_proportions_actual.tolist()],
            f"Predicted Proportions ({marker_name_for_output})" if marker_name_for_output else "Predicted Proportions": [round(p, 4) for p in pred_proportions_actual_list],
            f"MAE ({marker_name_for_output})" if marker_name_for_output else "MAE_Proportions": round(mae_sample, 4) if not np.isnan(mae_sample) else "N/A"
        }
        if marker_name_for_output: 
             entry[f"Predicted Logits ({marker_name_for_output})"] = [round(float(l), 4) for l in relevant_logits_np] if true_n > 0 else []
        results.append(entry)

    avg_overall_mae = np.mean([m for m in overall_mae_list if not np.isnan(m)]) if overall_mae_list else np.nan
    print(f"对于 {marker_name_for_output if marker_name_for_output else '集成模型'} 的所有样本的平均MAE (比例): {avg_overall_mae:.4f}" if not np.isnan(avg_overall_mae) else f"对于 {marker_name_for_output if marker_name_for_output else '集成模型'} 的所有样本的平均MAE (比例): N/A")
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    CSV_FILE_PATH = '附件2：不同混合比例的STR图谱数据Predict.csv'
    MAX_PEOPLE_FOR_RATIO_TASK_CONFIG = 6
    MAX_ALLELES_PER_MARKER_CONFIG = 0 
    MAX_SEQUENCE_LENGTH_CONFIG = 25 
    NUM_FEATURES_PER_ALLELE_EVENT_CONFIG = 1

    ATTENTION_D_MODEL_CONFIG = 128 
    LAMBDA_DECAY_CONFIG = 0.005 
    DENSE_UNITS_CONFIG = 256   
    DROPOUT_RATE_CONFIG = 0.0 
    LEARNING_RATE_CONFIG = 0.0001 
    BATCH_SIZE_CONFIG = 4 
    EPOCHS_CONFIG = 300 
    MODEL_SAVE_DIR = "trained_marker_ratio_models_jsd" 
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"--- 开始为每个Marker独立训练混合比例模型 (损失函数: Jensen-Shannon Divergence) ---")
    original_wide_df, sample_info_df_filtered = load_and_extract_ratio_info(
        CSV_FILE_PATH, MAX_PEOPLE_FOR_RATIO_TASK_CONFIG
    )
    if original_wide_df is None or sample_info_df_filtered is None or sample_info_df_filtered.empty:
        print("Failed"); exit()

    max_a = 0
    for col_name in original_wide_df.columns:
        match = re.match(r'(?:Allele|Size|Height)\s*(\d+)', col_name) 
        if match: max_a = max(max_a, int(match.group(1)))
    MAX_ALLELES_PER_MARKER_CONFIG = max_a
    if MAX_ALLELES_PER_MARKER_CONFIG == 0: print("错误: 未能从列名中确定最大等位基因对数量。"); exit()
    print(f"从CSV列名确定的最大等位基因/高度/大小对数: {MAX_ALLELES_PER_MARKER_CONFIG}")

    if 'Marker' not in original_wide_df.columns: print(f"错误: 原始DataFrame中缺少 'Marker' 列。"); exit()
        
    all_markers_unique = sorted(original_wide_df['Marker'].unique().tolist())
    NUM_MARKERS_CONFIG = len(all_markers_unique)
    print(f"将为以下 {NUM_MARKERS_CONFIG} 个Markers训练模型: {all_markers_unique}")
    print(f"注意力层输入序列长度 (max_sequence_length): {MAX_SEQUENCE_LENGTH_CONFIG}")

    trained_marker_models = {} 
    all_samples_data_for_markers = {} 
    print("\n--- 正在为所有Marker预处理输入数据 ---")
    for target_marker_name in all_markers_unique:
        print(f"为 Marker: {target_marker_name} 预处理数据...")
        X_features_sm, X_sizes_sm, X_masks_sm, X_N_sm, Y_targets_sm = preprocess_single_marker_inputs(
            original_wide_df, sample_info_df_filtered, target_marker_name,
            MAX_ALLELES_PER_MARKER_CONFIG, MAX_SEQUENCE_LENGTH_CONFIG, NUM_FEATURES_PER_ALLELE_EVENT_CONFIG
        )
        if X_features_sm is None or (isinstance(X_N_sm, np.ndarray) and X_N_sm.shape[0] == 0):
            print(f"为Marker '{target_marker_name}' 构建输入特征失败或无有效数据，跳过此Marker。")
            continue
        all_samples_data_for_markers[target_marker_name] = {
            "X_features": X_features_sm, "X_sizes": X_sizes_sm, "X_masks": X_masks_sm,
            "X_N_values": X_N_sm, "Y_targets": Y_targets_sm
        }
    
    for target_marker_name in all_markers_unique:
        if target_marker_name not in all_samples_data_for_markers:
            continue

        print(f"\n--- 正在为 Marker: {target_marker_name} 训练模型 ---")
        marker_data = all_samples_data_for_markers[target_marker_name]
        X_train_inputs_sm = [marker_data["X_features"], marker_data["X_sizes"], marker_data["X_masks"], marker_data["X_N_values"]]
        Y_train_sm = marker_data["Y_targets"]
        
        print(f"Marker '{target_marker_name}': 训练集样本数: {len(Y_train_sm)}")
        if len(Y_train_sm) == 0: print(f"Marker '{target_marker_name}' 训练集为空，跳过。"); continue

        model_prefix = target_marker_name.replace(" ", "_").replace(".", "_")
        single_marker_model = build_single_marker_ratio_model(
            MAX_SEQUENCE_LENGTH_CONFIG, NUM_FEATURES_PER_ALLELE_EVENT_CONFIG,
            MAX_PEOPLE_FOR_RATIO_TASK_CONFIG, ATTENTION_D_MODEL_CONFIG, LAMBDA_DECAY_CONFIG,
            DENSE_UNITS_CONFIG, DROPOUT_RATE_CONFIG, marker_name_prefix=model_prefix
        )
        single_marker_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_CONFIG), 
                                    loss=custom_jsd_loss)

        callbacks_sm = [
            EarlyStopping(monitor='loss', patience=75, verbose=1, restore_best_weights=True), 
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30, verbose=1, min_lr=1e-8) 
        ]
        print(f"开始训练 Marker: {target_marker_name}...")
        history = single_marker_model.fit(
            X_train_inputs_sm, Y_train_sm, 
            epochs=EPOCHS_CONFIG, batch_size=BATCH_SIZE_CONFIG, 
            callbacks=callbacks_sm, verbose=0 
        )
        
        model_filename = os.path.join(MODEL_SAVE_DIR, f"ratio_model_{model_prefix}.keras")
        try: 
            single_marker_model.save(model_filename)
            print(f"模型已保存到: {model_filename}")
            trained_marker_models[target_marker_name] = model_filename 
        except Exception as e: 
            print(f"保存Marker '{target_marker_name}' 模型时发生错误: {e}")

    print("\n\n--- 开始集成所有Marker模型的预测结果 ---")
    final_comparison_df = sample_info_df_filtered[['Sample File', 'N_Contributors', 'Y_Target_For_Loss']].copy()
    final_comparison_df.rename(columns={'Y_Target_For_Loss': 'True_Y_With_N'}, inplace=True)
    
    def extract_true_proportions(row): 
        true_n = int(row['N_Contributors'])
        true_y_with_n = row['True_Y_With_N']
        true_proportions_padded = true_y_with_n[:-1]
        return [round(p, 4) for p in (true_proportions_padded[:true_n] if true_n > 0 else [])]
    final_comparison_df['True Overall Proportions'] = final_comparison_df.apply(extract_true_proportions, axis=1)

    sample_marker_logits_collected = {sf: {} for sf in final_comparison_df['Sample File']}

    for marker_name, model_path in trained_marker_models.items():
        print(f"加载 Marker '{marker_name}' 的模型并获取其对所有样本的预测...")
        if marker_name not in all_samples_data_for_markers:
            print(f"未找到 Marker '{marker_name}' 的预处理数据，跳过。")
            continue
        try:
            model = load_model(model_path, custom_objects={
                'DistanceWeightedSelfAttention': DistanceWeightedSelfAttention,
                'custom_jsd_loss': custom_jsd_loss, 
                'CastToFloat32Layer': CastToFloat32Layer 
            })
        except Exception as e:
            print(f"加载模型 '{model_path}' 失败: {e}。跳过此Marker。")
            continue

        marker_data_for_pred = all_samples_data_for_markers[marker_name]
        inputs_for_marker_pred = [marker_data_for_pred["X_features"], 
                                  marker_data_for_pred["X_sizes"], 
                                  marker_data_for_pred["X_masks"], 
                                  marker_data_for_pred["X_N_values"]]
        
        if len(inputs_for_marker_pred[0]) == 0:
            print(f"Marker '{marker_name}' 没有数据进行预测。")
            continue

        marker_logits_all_samples = model.predict(inputs_for_marker_pred) 
        for idx, sample_file_iter in enumerate(sample_info_df_filtered['Sample File']):
            if idx < len(marker_logits_all_samples):
                 sample_marker_logits_collected[sample_file_iter][marker_name] = marker_logits_all_samples[idx]
        
        marker_pred_proportions_col = []
        marker_mae_col = []
        true_proportions_for_marker_col = [] 

        for idx_fc, row_fc in sample_info_df_filtered.iterrows(): 
            sample_file_fc = row_fc['Sample File']
            true_n_fc = row_fc['N_Contributors']
            true_y_with_n_fc = row_fc['Y_Target_For_Loss'] 
            true_proportions_padded_fc = true_y_with_n_fc[:-1]
            true_proportions_actual_fc = np.array(true_proportions_padded_fc[:true_n_fc] if true_n_fc > 0 else [], dtype=float)
            true_proportions_for_marker_col.append([round(p,4) for p in true_proportions_actual_fc.tolist()])

            pred_props_iter_list = []
            mae_marker_sample = np.nan
            if idx_fc < len(marker_logits_all_samples): 
                pred_logits_sample_iter = marker_logits_all_samples[idx_fc]
                if true_n_fc > 0:
                    relevant_logits_iter = pred_logits_sample_iter[:true_n_fc]
                    pred_props_iter_np = tf.nn.softmax(relevant_logits_iter).numpy()
                    pred_props_iter_list = [round(p,4) for p in pred_props_iter_np.tolist()]
                    if len(true_proportions_actual_fc) == len(pred_props_iter_np):
                        mae_marker_sample = mean_absolute_error(true_proportions_actual_fc, pred_props_iter_np)
            marker_pred_proportions_col.append(pred_props_iter_list)
            marker_mae_col.append(round(mae_marker_sample, 4) if not np.isnan(mae_marker_sample) else "N/A")

        if len(marker_pred_proportions_col) == len(final_comparison_df):
            final_comparison_df[f'Pred Proportions ({marker_name})'] = marker_pred_proportions_col
            final_comparison_df[f'MAE ({marker_name})'] = marker_mae_col
        else:
            print(f"警告: Marker '{marker_name}' 的预测列长度与DataFrame不匹配。")

    ensembled_proportions_col = []
    ensembled_mae_col = []

    for index, row in final_comparison_df.iterrows():
        sample_file = row['Sample File']
        true_n = row['N_Contributors']
        true_overall_proportions_list = row['True Overall Proportions']
        true_overall_proportions_np = np.array(true_overall_proportions_list, dtype=float)
        
        collected_logits_for_sample = []
        for marker_name_iter in all_markers_unique: 
            if marker_name_iter in sample_marker_logits_collected.get(sample_file, {}): 
                collected_logits_for_sample.append(sample_marker_logits_collected[sample_file][marker_name_iter])
        
        ensembled_pred_props_list = []
        ensembled_mae_sample = np.nan

        if collected_logits_for_sample and true_n > 0:
            avg_logits = np.mean(np.array(collected_logits_for_sample), axis=0) 
            relevant_avg_logits = avg_logits[:true_n]
            ensembled_pred_props_np = tf.nn.softmax(relevant_avg_logits).numpy()
            ensembled_pred_props_list = [round(p, 4) for p in ensembled_pred_props_np.tolist()]
            if len(true_overall_proportions_np) == len(ensembled_pred_props_np): 
                ensembled_mae_sample = mean_absolute_error(true_overall_proportions_np, ensembled_pred_props_np)
        
        ensembled_proportions_col.append(ensembled_pred_props_list)
        ensembled_mae_col.append(round(ensembled_mae_sample, 4) if not np.isnan(ensembled_mae_sample) else "N/A")

    final_comparison_df['Ensembled Predicted Proportions'] = ensembled_proportions_col
    final_comparison_df['Ensembled MAE'] = ensembled_mae_col
    
    cols_to_drop_for_print = ['True_Y_With_N']
    final_print_df = final_comparison_df.drop(columns=cols_to_drop_for_print, errors='ignore')

    first_cols = ['Sample File', 'N_Contributors', 'True Overall Proportions', 'Ensembled Predicted Proportions', 'Ensembled MAE']
    marker_cols = [col for col in final_print_df.columns if col not in first_cols]
    final_print_df = final_print_df[first_cols + sorted(marker_cols)]


    print("\n\n--- 最终集成预测结果与真实结果对比 (包含各Marker的独立预测) ---")
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', 2000) 
    print(final_print_df.to_string())

    final_comparison_filename = os.path.join(MODEL_SAVE_DIR, "final_ensembled_predictions_comparison.csv")
    try:
        final_print_df.to_csv(final_comparison_filename, index=False, encoding='utf-8-sig') 
        print(f"\n最终集成对比结果已保存到: {final_comparison_filename}")
    except Exception as e:
        print(f"保存最终集成对比结果时发生错误: {e}")

    avg_ensembled_mae_list = [m for m in ensembled_mae_col if isinstance(m, (float,int)) and not np.isnan(m)]
    avg_ensembled_mae = np.mean(avg_ensembled_mae_list) if avg_ensembled_mae_list else np.nan
    print(f"\n所有样本的平均集成MAE (比例): {avg_ensembled_mae:.4f}" if not np.isnan(avg_ensembled_mae) else "所有样本的平均集成MAE (比例): N/A")

    print("\n--- 所有流程执行完毕 ---")
