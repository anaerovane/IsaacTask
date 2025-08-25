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
import ast 
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
        if match:
            parsed_ratios.append(match.group(1))
        else:
            parsed_ratios.append(None)
    return parsed_ratios

def convert_ratio_to_proportions(ratio_str, num_contributors):
    if ratio_str is None:
        return None
    try:
        parts = [float(p) for p in ratio_str.split(';')]
        if len(parts) != num_contributors:
            if num_contributors == 0 and not parts:
                return []
            if num_contributors == 1 and len(parts) == 1:
                pass
            else:
                return None
        if num_contributors == 0:
            return []
        total = sum(parts)
        if total == 0:
            return [0.0] * num_contributors
        return [p / total for p in parts]
    except:
        return None

def load_and_extract_info(csv_file_path):
    df = pd.read_csv(csv_file_path)
    REQUIRED_COLS = ['Sample File', 'Predicted Number of People', 'Allele_Counts_From_Contributors', 'Marker']
    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少必需的列 '{col}'。请检查列名是否与代码中定义的 '{col}' 完全一致（包括大小写）。")
            return None, None

    df.loc[:, 'Predicted Number of People'] = pd.to_numeric(df['Predicted Number of People'], errors='coerce').fillna(0).astype(int)
    df.loc[:, 'N_Contributors'] = df['Predicted Number of People']
    df.loc[:, 'Allele_Counts_Dict'] = df['Allele_Counts_From_Contributors'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else {}
    )
    sample_info_df = df[['Sample File', 'Predicted Number of People']].drop_duplicates().reset_index(drop=True)
    sample_info_df.loc[:, 'N_Contributors'] = sample_info_df['Predicted Number of People']
    sample_info_df.loc[:, 'Raw_Ratio_Str'] = parse_ratio_from_sample_file_name(sample_info_df['Sample File'])
    df_filtered_for_allele_counts = df[
        (df['N_Contributors'] >= 0) & 
        (df['Allele_Counts_Dict'].apply(lambda x: isinstance(x, dict)))
    ].copy()

    return df_filtered_for_allele_counts, sample_info_df

def preprocess_single_marker_inputs(
    original_marker_df,
    sample_info_df_for_model_samples,
    target_marker_name,
    max_alleles_per_marker_from_cols,
    num_features_per_allele_event=1
    ):

    sample_files_to_process = sample_info_df_for_model_samples['Sample File'].tolist()

    X_marker_features_single, X_marker_allele_sizes_single, X_marker_masks_single = [], [], []
    X_N_values_list, Y_targets_allele_proportions_list = [], [] 

    all_sizes_marker_specific, all_heights_marker_specific = [], []
    
    is_first_sample_for_debug = True 

    if original_marker_df.empty or 'Marker' not in original_marker_df.columns or target_marker_name not in original_marker_df['Marker'].unique():
        return None, None, None, None, None, None

    all_true_allele_names_for_marker = set()
    for _, row in original_marker_df.iterrows():
        current_allele_counts_dict = row['Allele_Counts_Dict']
        for key in current_allele_counts_dict.keys():
            all_true_allele_names_for_marker.add(str(key))
    
    canonical_allele_names_for_marker = sorted(list(all_true_allele_names_for_marker))
 
    marker_specific_sequence_length = len(canonical_allele_names_for_marker)
    
    if marker_specific_sequence_length == 0:
        print(f"警告: Marker '{target_marker_name}' 没有在 Allele_Counts_From_Contributors 中找到任何真实等位基因。跳过此Marker的数据预处理。")
        return None, None, None, None, None, None


    allele_name_to_target_index_map = {name: i for i, name in enumerate(canonical_allele_names_for_marker)}


    for idx, row in original_marker_df.iterrows():
        for i in range(1, max_alleles_per_marker_from_cols + 1):
            size_col = f'Size {i}' if f'Size {i}' in row.index else f'Size{i}'
            allele_col = f'Allele {i}' if f'Allele {i}' in row.index else f'Allele{i}'
            height_col = f'Height {i}' if f'Height {i}' in row.index else f'Height{i}'
            
            if size_col in row.index and allele_col in row.index and height_col in row.index:
                size_val = pd.to_numeric(row.get(size_col), errors='coerce')
                allele_val = row.get(allele_col)
                height_val = pd.to_numeric(row.get(height_col), errors='coerce')
                
                if pd.notna(size_val) and pd.notna(height_val) and height_val > 0 and pd.notna(allele_val):
                    all_sizes_marker_specific.append(float(size_val))
                    all_heights_marker_specific.append(float(height_val))
        
    size_scaler, height_scaler = MinMaxScaler(), MinMaxScaler()
    if all_sizes_marker_specific:
        size_scaler.fit(np.array(all_sizes_marker_specific).reshape(-1, 1))
    if all_heights_marker_specific:
        height_scaler.fit(np.array(all_heights_marker_specific).reshape(-1, 1))

    for sample_idx, sample_file_name in enumerate(sample_files_to_process):
        marker_row_df_for_sample = original_marker_df[original_marker_df['Sample File'] == sample_file_name]

        sample_n_info = sample_info_df_for_model_samples[sample_info_df_for_model_samples['Sample File'] == sample_file_name]
        if sample_n_info.empty:
            X_marker_features_single.append(np.zeros((marker_specific_sequence_length, num_features_per_allele_event)))
            X_marker_allele_sizes_single.append(np.zeros(marker_specific_sequence_length))
            X_marker_masks_single.append(np.zeros(marker_specific_sequence_length, dtype=bool))
            X_N_values_list.append([0.0])
            Y_targets_allele_proportions_list.append(np.zeros(marker_specific_sequence_length)) 
            continue
        current_n_contributors = sample_n_info.iloc[0]['N_Contributors']
        X_N_values_list.append([float(current_n_contributors)])

        if marker_row_df_for_sample.empty:
            X_marker_features_single.append(np.zeros((marker_specific_sequence_length, num_features_per_allele_event)))
            X_marker_allele_sizes_single.append(np.zeros(marker_specific_sequence_length))
            X_marker_masks_single.append(np.zeros(marker_specific_sequence_length, dtype=bool))
            Y_targets_allele_proportions_list.append(np.zeros(marker_specific_sequence_length)) 
            continue

        current_marker_data_series = marker_row_df_for_sample.iloc[0]

        observed_alleles_in_sample_for_X = {} 
        for i in range(1, max_alleles_per_marker_from_cols + 1):
            size_col = f'Size {i}' if f'Size {i}' in current_marker_data_series.index else f'Size{i}'
            allele_col = f'Allele {i}' if f'Allele {i}' in current_marker_data_series.index else f'Allele{i}'
            height_col = f'Height {i}' if f'Height {i}' in current_marker_data_series.index else f'Height{i}'
            if size_col in current_marker_data_series.index and allele_col in current_marker_data_series.index and height_col in current_marker_data_series.index:
                size_val = pd.to_numeric(current_marker_data_series.get(size_col), errors='coerce')
                allele_val = current_marker_data_series.get(allele_col)
                height_val = pd.to_numeric(current_marker_data_series.get(height_col), errors='coerce')
                
                if pd.notna(size_val) and pd.notna(height_val) and height_val > 0 and pd.notna(allele_val):
                    observed_alleles_in_sample_for_X[str(allele_val)] = {'Size': float(size_val), 'Height': float(height_val)}
        
        padded_features = np.zeros((marker_specific_sequence_length, num_features_per_allele_event))
        padded_sizes = np.zeros(marker_specific_sequence_length)
        mask = np.zeros(marker_specific_sequence_length, dtype=bool)

        current_sample_allele_names_observed_for_debug = [] 
        current_sample_sizes_raw_for_debug = [] 

        for i, canonical_allele_name in enumerate(canonical_allele_names_for_marker):
            if canonical_allele_name in observed_alleles_in_sample_for_X:
                obs_data = observed_alleles_in_sample_for_X[canonical_allele_name]
                current_sample_allele_names_observed_for_debug.append(canonical_allele_name)
                current_sample_sizes_raw_for_debug.append(obs_data['Size'])
                norm_size = size_scaler.transform(np.array([[obs_data['Size']]])).flatten()[0] if all_sizes_marker_specific else obs_data['Size']
                norm_height = height_scaler.transform(np.array([[obs_data['Height']]])).flatten()[0] if all_heights_marker_specific else obs_data['Height']
                padded_features[i, 0] = norm_height
                padded_sizes[i] = norm_size
                mask[i] = True

        X_marker_features_single.append(padded_features)
        X_marker_allele_sizes_single.append(padded_sizes)
        X_marker_masks_single.append(mask)

        current_allele_counts_dict = current_marker_data_series['Allele_Counts_Dict']
        temp_allele_counts = np.zeros(marker_specific_sequence_length, dtype=float) 
        filled_count = 0
        missing_keys = []

        for allele_name_from_dict_key, count_val in current_allele_counts_dict.items():
            str_allele_name_from_dict_key = str(allele_name_from_dict_key)
            
            if str_allele_name_from_dict_key in allele_name_to_target_index_map:
                target_index = allele_name_to_target_index_map[str_allele_name_from_dict_key]
                if target_index < marker_specific_sequence_length:
                    temp_allele_counts[target_index] = float(count_val)
                    filled_count += 1
                else:
                    missing_keys.append(f"'{str_allele_name_from_dict_key}' (index out of bounds for marker_specific_sequence_length)")
            else:
                missing_keys.append(f"'{str_allele_name_from_dict_key}' (not in canonical map for this marker)")

        total_counts_for_proportions = np.sum(temp_allele_counts)
        if total_counts_for_proportions > 0:
            padded_allele_proportions = temp_allele_counts / total_counts_for_proportions
        else:
            padded_allele_proportions = np.zeros(marker_specific_sequence_length, dtype=float)  

        Y_targets_allele_proportions_list.append(padded_allele_proportions)

        if is_first_sample_for_debug:
            print(f"\n--- Debug: Marker '{target_marker_name}' - First Sample '{sample_file_name}' ---")
            print(f"  Marker Specific Sequence Length: {marker_specific_sequence_length}")
            print(f"  Raw Allele_Counts_From_Contributors string: {current_marker_data_series.get('Allele_Counts_From_Contributors', 'N/A')}")
            print(f"  Parsed Allele_Counts_Dict: {current_allele_counts_dict}")
            print(f"  Canonical Allele Names for Marker (Y-target order): {canonical_allele_names_for_marker}")
            print(f"  Allele Name to Target Index Map: {allele_name_to_target_index_map}")
            print(f"  Observed Allele Names (from 'Allele X' column for this sample): {current_sample_allele_names_observed_for_debug}")
            print(f"  Observed Allele Sizes (from 'Size X' column for this sample): {current_sample_sizes_raw_for_debug}")
            print(f"  Is Allele_Counts_Dict empty? {not bool(current_allele_counts_dict)}")
            print(f"  Number of allele counts filled into target vector: {filled_count} / {len(current_allele_counts_dict)}")
            if missing_keys:
                print(f"  Allele_Counts_Dict keys NOT mapped to target vector: {missing_keys}")
            print(f"  Resulting padded_allele_proportions for this sample (first 5 values): {padded_allele_proportions[:min(5, marker_specific_sequence_length)].tolist()}")
            print(f"  Resulting padded_features for this sample (first 2 rows, first feature): {padded_features[:min(2, marker_specific_sequence_length), 0].tolist()}")
            print(f"  Resulting padded_sizes for this sample (first 2 values): {padded_sizes[:min(2, marker_specific_sequence_length)].tolist()}")
            print(f"  Resulting mask for this sample (first 5 values): {mask[:min(5, marker_specific_sequence_length)].tolist()}")
            is_first_sample_for_debug = False
       
    if Y_targets_allele_proportions_list:
        print(f"\n--- Marker: {target_marker_name} 的真实目标向量 (Y_targets_sm) ---")
        print(np.array(Y_targets_allele_proportions_list)[:min(5, len(Y_targets_allele_proportions_list))])
    else:
        print(f"\n--- Marker: {target_marker_name} 没有有效的真实目标向量 (Y_targets_sm) ---")

    return (np.array(X_marker_features_single), np.array(X_marker_allele_sizes_single),
            np.array(X_marker_masks_single), np.array(X_N_values_list), np.array(Y_targets_allele_proportions_list),
            marker_specific_sequence_length) 

def custom_jsd_loss(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_true_safe = y_true + epsilon
    y_pred_safe = y_pred + epsilon
    y_true_norm = y_true_safe / tf.reduce_sum(y_true_safe, axis=-1, keepdims=True)
    y_pred_norm = y_pred_safe / tf.reduce_sum(y_pred_safe, axis=-1, keepdims=True)
    M = 0.5 * (y_true_norm + y_pred_norm)
    M_safe = M / tf.reduce_sum(M, axis=-1, keepdims=True) 
    kl_p_m = tf.keras.losses.kullback_leibler_divergence(y_true_norm, M_safe)
    kl_q_m = tf.keras.losses.kullback_leibler_divergence(y_pred_norm, M_safe)
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd


def build_single_marker_proportion_model( 
    max_sequence_length, 
    num_features_per_allele_event,
    attention_d_model=128, lambda_decay_attention=0.01,
    dense_units=256, dropout_rate=0.0, marker_name_prefix="marker"
):
    marker_feature_input = Input(shape=(max_sequence_length, num_features_per_allele_event), name=f'{marker_name_prefix}_features_input')
    marker_allele_size_input = Input(shape=(max_sequence_length,), name=f'{marker_name_prefix}_allele_sizes_input')
    marker_mask_input = Input(shape=(max_sequence_length,), dtype=tf.bool, name=f'{marker_name_prefix}_mask_input')
    num_contributors_input = Input(shape=(1,), name='num_contributors_input', dtype='float32')

    attention_layer = DistanceWeightedSelfAttention(d_model=attention_d_model, lambda_decay=lambda_decay_attention, name=f'{marker_name_prefix}_attention')
    attention_output_sequence = attention_layer(marker_feature_input, marker_allele_size_input, mask=marker_mask_input)
    pooled_attention_output = GlobalAveragePooling1D(name=f'{marker_name_prefix}_gap')(attention_output_sequence, mask=marker_mask_input)

    combined_features = Concatenate(name=f'{marker_name_prefix}_combine_with_N')([pooled_attention_output, num_contributors_input])

    dense_layer = Dense(dense_units, activation='relu', name=f'{marker_name_prefix}_dense_1')(combined_features)
    if dropout_rate > 0: dense_layer = Dropout(dropout_rate, name=f'{marker_name_prefix}_dropout_1')(dense_layer)
    dense_layer = Dense(dense_units // 2, activation='relu', name=f'{marker_name_prefix}_dense_2')(dense_layer)
    if dropout_rate > 0: dense_layer = Dropout(dropout_rate, name=f'{marker_name_prefix}_dropout_2')(dense_layer)
    dense_layer = Dense(dense_units // 4, activation='relu', name=f'{marker_name_prefix}_dense_3')(dense_layer)
    allele_proportions_predictions = Dense(max_sequence_length, activation='softmax', name=f'{marker_name_prefix}_allele_proportions_predictions')(dense_layer)

    model_inputs = [marker_feature_input, marker_allele_size_input, marker_mask_input, num_contributors_input]
    model = Model(inputs=model_inputs, outputs=allele_proportions_predictions)
    return model

def get_predictions_and_compare(model, X_inputs_list, Y_targets_all_samples, X_masks_all_samples, sample_files_all_samples, marker_name_for_output="", dataset_type=""):
    predictions_raw = model.predict(X_inputs_list, verbose=0) 
    results = []
    overall_mae_list = []

    for i in range(len(sample_files_all_samples)):
        sample_file = sample_files_all_samples[i]
        true_allele_proportions = Y_targets_all_samples[i] 
        pred_allele_proportions = predictions_raw[i] 
        current_mask = X_masks_all_samples[i] 
        true_masked = true_allele_proportions[current_mask]
        pred_masked = pred_allele_proportions[current_mask]
        mae_sample = np.nan
        if len(true_masked) > 0 and len(pred_masked) == len(true_masked):
            mae_sample = mean_absolute_error(true_masked, pred_masked)
            overall_mae_list.append(mae_sample)

        entry = {
            "Sample File": sample_file,
            f"True Allele Proportions ({marker_name_for_output})" if marker_name_for_output else "True Allele Proportions": [round(c, 4) for c in true_masked.tolist()],
            f"Predicted Allele Proportions ({marker_name_for_output})" if marker_name_for_output else "Predicted Allele Proportions": [round(c, 4) for c in pred_masked.tolist()],
            f"MAE ({marker_name_for_output})" if marker_name_for_output else "MAE_Allele_Proportions": round(mae_sample, 4) if not np.isnan(mae_sample) else "N/A"
        }
        results.append(entry)

    avg_overall_mae = np.mean([m for m in overall_mae_list if not np.isnan(m)]) if overall_mae_list else np.nan
    print(f"对于 {marker_name_for_output} 的 {dataset_type} 平均MAE (等位基因比例): {avg_overall_mae:.4f}" if not np.isnan(avg_overall_mae) else f"对于 {marker_name_for_output} 的 {dataset_type} 平均MAE (等位基因比例): N/A")

    if len(results) > 0:
        print(f"  Debug: First sample's True Masked Proportions for {marker_name_for_output}: {results[0].get(f'True Allele Proportions ({marker_name_for_output})', 'N/A')}")
        print(f"  Debug: First sample's Pred Masked Proportions for {marker_name_for_output}: {results[0].get(f'Predicted Allele Proportions ({marker_name_for_output})', 'N/A')}")
    else:
        print(f"  Debug: No results generated for {marker_name_for_output} in get_predictions_and_compare.")

    return pd.DataFrame(results)

def load_contributor_genotypes(genotype_csv_path):
    try:
        df_genotype = pd.read_csv(genotype_csv_path)
        print(f"文件 '{genotype_csv_path}' 加载成功。")
    except FileNotFoundError:
        print(f"错误: 基因型文件 '{genotype_csv_path}' 未找到。")
        return None
    except Exception as e:
        print(f"加载基因型文件 '{genotype_csv_path}' 时发生错误: {e}")
        return None

    required_id_cols = ['Sample ID'] 
    for col in required_id_cols:
        if col not in df_genotype.columns:
            print(f"错误: 基因型文件中缺少必需的列 '{col}'。")
            return None

    contributor_genotypes = {}
    marker_cols = [col for col in df_genotype.columns if col not in ['Reseach', 'ID', 'Sample ID']]

    if not marker_cols:
        print("错误: 基因型文件中未找到任何 Marker 列。请确保 Marker 列存在。")
        return None

    for _, row in df_genotype.iterrows():
        person_id = str(row['Sample ID']) 

        if person_id not in contributor_genotypes:
            contributor_genotypes[person_id] = {}
        
        for marker_col in marker_cols:
            marker_name = marker_col 
            alleles_str = row.get(marker_col) 

            if pd.notna(alleles_str) and isinstance(alleles_str, str):
                alleles = [a.strip() for a in alleles_str.split(',') if a.strip()]
                if len(alleles) == 2: 
                    contributor_genotypes[person_id][marker_name] = sorted(alleles)
                elif len(alleles) == 1: 
                    contributor_genotypes[person_id][marker_name] = sorted([alleles[0], alleles[0]])
                else:
                    contributor_genotypes[person_id][marker_name] = [] 
            else:
                contributor_genotypes[person_id][marker_name] = [] 
    print(f"加载了 {len(contributor_genotypes)} 位贡献者的基因型数据。")
    return contributor_genotypes

def infer_proportions_and_score(
    predicted_allele_proportions_by_marker, 
    combo_person_ids, 
    contributor_genotypes_db, 
    all_markers_unique 
):
    
    if not SCIPY_AVAILABLE:
        inferred_proportions = np.ones(len(combo_person_ids)) / len(combo_person_ids)
        total_rmse = 0.0
        total_alleles_compared = 0

        for marker_name, pred_props_dict in predicted_allele_proportions_by_marker.items():
            if marker_name not in all_markers_unique: continue 
            
            theoretical_props_marker = {}
            for person_id in combo_person_ids:
                if marker_name in contributor_genotypes_db.get(person_id, {}):
                    for allele in contributor_genotypes_db[person_id][marker_name]:
                        theoretical_props_marker[str(allele)] = theoretical_props_marker.get(str(allele), 0) + (1.0 / (len(combo_person_ids) * 2)) 

            for allele_name, predicted_prop in pred_props_dict.items():
                theoretical_prop = theoretical_props_marker.get(allele_name, 0.0)
                total_rmse += (predicted_prop - theoretical_prop)**2
                total_alleles_compared += 1
        
        if total_alleles_compared > 0:
            rmse_score = np.sqrt(total_rmse / total_alleles_compared)
        else:
            rmse_score = float('inf') 
        
        return inferred_proportions.tolist(), rmse_score

    all_alleles_in_system = set()
    for marker in all_markers_unique:
        if marker in predicted_allele_proportions_by_marker: 
            for allele_name in predicted_allele_proportions_by_marker[marker].keys():
                all_alleles_in_system.add(allele_name)
        for person_id in combo_person_ids:
            if marker in contributor_genotypes_db.get(person_id, {}):
                for allele in contributor_genotypes_db[person_id][marker]:
                    all_alleles_in_system.add(str(allele)) 
    sorted_alleles_in_system = sorted(list(all_alleles_in_system))
    allele_to_row_idx = {allele: i for i, allele in enumerate(sorted_alleles_in_system)}
    
    num_alleles = len(sorted_alleles_in_system)
    num_persons = len(combo_person_ids)

    if num_alleles == 0:
        return np.zeros(num_persons).tolist(), float('inf') 

    A = np.zeros((num_alleles, num_persons)) 
    b = np.zeros(num_alleles) 
    for marker, pred_props_dict in predicted_allele_proportions_by_marker.items():
        for allele_name, predicted_prop in pred_props_dict.items():
            if allele_name in allele_to_row_idx:
                b[allele_to_row_idx[allele_name]] += predicted_prop

    total_predicted_prop_in_b = np.sum(b)
    if total_predicted_prop_in_b > 0:
        b /= total_predicted_prop_in_b
    else:
        pass 
    person_total_alleles_in_relevant_markers = np.zeros(num_persons)
    for person_idx, person_id in enumerate(combo_person_ids):
        for marker in all_markers_unique: 
            if marker in contributor_genotypes_db.get(person_id, {}):
                person_total_alleles_in_relevant_markers[person_idx] += len(contributor_genotypes_db[person_id][marker])  
    
    for person_idx, person_id in enumerate(combo_person_ids):
        for marker in all_markers_unique:
            if marker in contributor_genotypes_db.get(person_id, {}):
                for allele_in_genotype in contributor_genotypes_db[person_id][marker]:
                    str_allele_in_genotype = str(allele_in_genotype)
                    if str_allele_in_genotype in allele_to_row_idx:
                        if person_total_alleles_in_relevant_markers[person_idx] > 0:
                            A[allele_to_row_idx[str_allele_in_genotype], person_idx] += 1.0 / person_total_alleles_in_relevant_markers[person_idx]

    result = lsq_linear(A, b, bounds=(0, np.inf))  
    inferred_proportions_arr = result.x
    
    prop_sum = np.sum(inferred_proportions_arr)
    if prop_sum > 0:
        inferred_proportions_arr /= prop_sum
    else:
        inferred_proportions_arr = np.zeros(num_persons)
    
    theoretical_proportions_b = A @ inferred_proportions_arr
    rmse_score = np.sqrt(np.sum((b - theoretical_proportions_b)**2))
    
    return inferred_proportions_arr.tolist(), rmse_score

if __name__ == '__main__':
    CSV_FILE_PATH = 'p3k1_with_prediction.csv'
    MAX_PEOPLE_FOR_RATIO_TASK_CONFIG = 6 
    MAX_ALLELES_PER_MARKER_FROM_COLS = 0 
    NUM_FEATURES_PER_ALLELE_EVENT_CONFIG = 1

    ATTENTION_D_MODEL_CONFIG = 128
    LAMBDA_DECAY_CONFIG = 0.005
    DENSE_UNITS_CONFIG = 256
    DROPOUT_RATE_CONFIG = 0.0
    LEARNING_RATE_CONFIG = 0.0001
    BATCH_SIZE_CONFIG = 4
    EPOCHS_CONFIG = 750
    MODEL_SAVE_DIR = "trained_marker_proportion_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    tf.keras.utils.get_custom_objects().update({
        'DistanceWeightedSelfAttention': DistanceWeightedSelfAttention,
        'CastToFloat32Layer': CastToFloat32Layer,
        'custom_jsd_loss': custom_jsd_loss
    })
    try:
        temp_df = pd.read_csv(CSV_FILE_PATH)
        max_a = 0
        for col_name in temp_df.columns:
            match = re.match(r'(?:Allele|Size|Height)\s*(\d+)', col_name)
            if match: max_a = max(max_a, int(match.group(1)))
        MAX_ALLELES_PER_MARKER_FROM_COLS = max_a
        if MAX_ALLELES_PER_MARKER_FROM_COLS == 0:
            print("错误: 未能从列名中确定最大等位基因对数量。请确保存在 'Size X', 'Allele X', 'Height X' 格式的列。"); exit()
        
        print(f"从CSV列名确定的最大等位基因/高度/大小对数 (MAX_ALLELES_PER_MARKER_FROM_COLS): {MAX_ALLELES_PER_MARKER_FROM_COLS}")

    except Exception as e:
        print(f"读取CSV以确定最大等位基因对数量时发生错误: {e}"); exit()
    
    original_df_with_allele_counts, sample_info_df = load_and_extract_info(CSV_FILE_PATH)

    if original_df_with_allele_counts is None or sample_info_df is None or sample_info_df.empty:
        print("数据加载或初步过滤失败，程序退出。请检查CSV文件内容和列名。"); exit()

    all_markers_unique = sorted(original_df_with_allele_counts['Marker'].unique().tolist())
    NUM_MARKERS_CONFIG = len(all_markers_unique)

    trained_marker_models = {}
    all_samples_data_for_markers = {}

    for target_marker_name in all_markers_unique:
        marker_subset_df = original_df_with_allele_counts[original_df_with_allele_counts['Marker'] == target_marker_name]

        X_features_full, X_sizes_full, X_masks_full, X_N_full, Y_targets_full, marker_specific_seq_len = preprocess_single_marker_inputs(
            marker_subset_df,
            sample_info_df,
            target_marker_name,
            MAX_ALLELES_PER_MARKER_FROM_COLS,
            NUM_FEATURES_PER_ALLELE_EVENT_CONFIG
        )

        if X_features_full is None or (isinstance(X_N_full, np.ndarray) and X_N_full.shape[0] == 0):
            continue
        
        all_samples_data_for_markers[target_marker_name] = {
            "X_full_features": X_features_full, "X_full_sizes": X_sizes_full, "X_full_masks": X_masks_full,
            "X_full_N_values": X_N_full, "Y_full_targets": Y_targets_full, "full_sample_files": sample_info_df['Sample File'].tolist(),
            "marker_specific_sequence_length": marker_specific_seq_len
        }

    for target_marker_name in all_markers_unique:
        if target_marker_name not in all_samples_data_for_markers:
            continue

        print(f"\n--- 正在训练 Marker: {target_marker_name} 模型 ---")
        marker_data = all_samples_data_for_markers[target_marker_name]
        
        X_train_inputs_sm = [marker_data["X_full_features"], marker_data["X_full_sizes"], marker_data["X_full_masks"], marker_data["X_full_N_values"]]
        Y_train_sm = marker_data["Y_full_targets"]

        X_val_inputs_sm = X_train_inputs_sm
        Y_val_sm = Y_train_sm
        
        print(f"Marker '{target_marker_name}': 完整数据集样本数: {len(Y_train_sm)}")
        if len(Y_train_sm) == 0:
            print(f"Marker '{target_marker_name}' 训练集为空，跳过训练。")
            continue

        model_prefix = target_marker_name.replace(" ", "_").replace(".", "_")
        single_marker_model = build_single_marker_proportion_model(
            marker_data["marker_specific_sequence_length"],
            NUM_FEATURES_PER_ALLELE_EVENT_CONFIG,
            ATTENTION_D_MODEL_CONFIG, LAMBDA_DECAY_CONFIG,
            DENSE_UNITS_CONFIG, DROPOUT_RATE_CONFIG, marker_name_prefix=model_prefix
        )
        single_marker_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_CONFIG),
                                    loss=custom_jsd_loss)

        callbacks_sm = [
            EarlyStopping(monitor='val_loss', patience=75, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1, min_lr=1e-8)
        ]
        print(f"开始训练 Marker: {target_marker_name}...")
        history = single_marker_model.fit(
            X_train_inputs_sm, Y_train_sm,
            epochs=EPOCHS_CONFIG, batch_size=BATCH_SIZE_CONFIG,
            callbacks=callbacks_sm, verbose=1,
            validation_data=(X_val_inputs_sm, Y_val_sm)
        )

        model_filename = os.path.join(MODEL_SAVE_DIR, f"proportion_model_{model_prefix}.keras")
        try:
            single_marker_model.save(model_filename)
            print(f"模型已保存到: {model_filename}")
        except Exception as e:
            print(f"保存Marker '{target_marker_name}' 模型时发生错误: {e}")
        
        print(f"\n--- Marker: {target_marker_name} 模型评估 ---")
        get_predictions_and_compare(
            single_marker_model, X_train_inputs_sm, Y_train_sm, marker_data["X_full_masks"],
            marker_data["full_sample_files"], marker_name_for_output=target_marker_name, dataset_type="完整数据集"
        )


    print("\n--- 最终集成预测结果与真实结果对比 (包含各Marker的独立预测) ---")
    final_comparison_df = sample_info_df[['Sample File', 'N_Contributors']].copy()
    final_comparison_df = final_comparison_df.sort_values(by='Sample File').reset_index(drop=True)

    all_marker_results_flat = {} 

    for marker_name in all_markers_unique:
        if marker_name not in all_samples_data_for_markers:
            continue
        
        model_path = os.path.join(MODEL_SAVE_DIR, f"proportion_model_{marker_name.replace(' ', '_').replace('.', '_')}.keras")
        try:
            model = load_model(model_path, custom_objects={'custom_jsd_loss': custom_jsd_loss})
        except Exception as e:
            print(f"加载模型 '{model_path}' 失败: '{e}'。跳过此Marker。")
            continue

        marker_data_for_pred = all_samples_data_for_markers[marker_name]
        X_full_inputs_for_pred = [marker_data_for_pred["X_full_features"],
                                  marker_data_for_pred["X_full_sizes"],
                                  marker_data_for_pred["X_full_masks"],
                                  marker_data_for_pred["X_full_N_values"]]
        Y_full_targets_for_pred = marker_data_for_pred["Y_full_targets"]
        X_full_masks_for_pred = marker_data_for_pred["X_full_masks"]
        full_sample_files_for_pred = marker_data_for_pred["full_sample_files"]

        if len(X_full_inputs_for_pred[0]) == 0:
            continue

        marker_pred_proportions_all_samples = model.predict(X_full_inputs_for_pred, verbose=0)

        for idx_pred, sample_file_pred in enumerate(full_sample_files_for_pred):
            true_allele_proportions_for_sample = Y_full_targets_for_pred[idx_pred]
            current_mask = X_full_masks_for_pred[idx_pred]
            pred_proportions_sample_iter = marker_pred_proportions_all_samples[idx_pred]

            true_masked = true_allele_proportions_for_sample[current_mask].tolist()
            pred_masked = pred_proportions_sample_iter[current_mask].tolist()

            mae_marker_sample = np.nan
            if len(true_masked) > 0 and len(pred_masked) == len(true_masked):
                mae_marker_sample = mean_absolute_error(true_masked, pred_masked)
            
            all_marker_results_flat[(sample_file_pred, marker_name)] = {
                'True Proportions': [round(c, 4) for c in true_masked],
                'Pred Proportions': [round(c, 4) for c in pred_masked],
                'MAE': round(mae_marker_sample, 4) if not np.isnan(mae_marker_sample) else "N/A"
            }

    final_results_rows = []
    for index, row in sample_info_df.sort_values(by='Sample File').reset_index(drop=True).iterrows():
        sample_file = row['Sample File']
        current_row_dict = {
            'Sample File': sample_file,
            'N_Contributors': row['N_Contributors']
        }
        
        ensembled_true_proportions_for_sample = {}
        ensembled_pred_proportions_for_sample = {}

        for marker_name in all_markers_unique:
            result_key = (sample_file, marker_name)
            if result_key in all_marker_results_flat:
                marker_results = all_marker_results_flat[result_key]
                current_row_dict[f'True Allele Proportions ({marker_name})'] = marker_results['True Proportions']
                current_row_dict[f'Pred Allele Proportions ({marker_name})'] = marker_results['Pred Proportions']
                current_row_dict[f'MAE ({marker_name})'] = marker_results['MAE']

                true_proportions_list = marker_results['True Proportions']
                pred_proportions_list = marker_results['Pred Proportions']

                marker_row_for_sample_data = original_df_with_allele_counts[
                    (original_df_with_allele_counts['Sample File'] == sample_file) &
                    (original_df_with_allele_counts['Marker'] == marker_name)
                ]
                
                if not marker_row_for_sample_data.empty:
                    current_allele_counts_dict = marker_row_for_sample_data.iloc[0]['Allele_Counts_Dict']
                    
                    temp_unique_alleles_for_agg = set()
                    for key in current_allele_counts_dict.keys():
                        temp_unique_alleles_for_agg.add(str(key))
                    for i in range(1, MAX_ALLELES_PER_MARKER_FROM_COLS + 1):
                        allele_col = f'Allele {i}' if f'Allele {i}' in marker_row_for_sample_data.iloc[0].index else f'Allele{i}'
                        if allele_col in marker_row_for_sample_data.iloc[0].index:
                            allele_val = marker_row_for_sample_data.iloc[0].get(allele_col)
                            if pd.notna(allele_val):
                                temp_unique_alleles_for_agg.add(str(allele_val))
                    
                    canonical_allele_names_for_marker_agg = sorted(list(temp_unique_alleles_for_agg))
                    marker_specific_seq_len_for_agg = all_samples_data_for_markers[marker_name]["marker_specific_sequence_length"]
                    canonical_allele_names_for_marker_agg = canonical_allele_names_for_marker_agg[:marker_specific_seq_len_for_agg]
                    
                    for i, allele_name in enumerate(canonical_allele_names_for_marker_agg):
                        if i < len(true_proportions_list) and i < len(pred_proportions_list):
                            ensembled_true_proportions_for_sample[allele_name] = true_proportions_list[i]
                            ensembled_pred_proportions_for_sample[allele_name] = pred_proportions_list[i]
            else:
                current_row_dict[f'True Allele Proportions ({marker_name})'] = []
                current_row_dict[f'Pred Allele Proportions ({marker_name})'] = []
                current_row_dict[f'MAE ({marker_name})'] = "N/A"

        sorted_alleles_for_ensemble = sorted(ensembled_true_proportions_for_sample.keys())
        ensembled_true_proportions_list_sorted = [ensembled_true_proportions_for_sample[a] for a in sorted_alleles_for_ensemble]
        ensembled_pred_proportions_list_sorted = [ensembled_pred_proportions_for_sample[a] for a in sorted_alleles_for_ensemble]
        
        ensembled_mae_sample = np.nan
        if len(ensembled_true_proportions_list_sorted) > 0 and len(ensembled_pred_proportions_list_sorted) == len(ensembled_true_proportions_list_sorted):
            ensembled_mae_sample = mean_absolute_error(ensembled_true_proportions_list_sorted, ensembled_pred_proportions_list_sorted)
        
        current_row_dict['True Overall Proportions'] = ensembled_true_proportions_list_sorted
        current_row_dict['Ensembled Predicted Proportions'] = ensembled_pred_proportions_list_sorted
        current_row_dict['Ensembled MAE'] = round(ensembled_mae_sample, 4) if not np.isnan(ensembled_mae_sample) else "N/A"
        
        final_results_rows.append(current_row_dict)

    final_print_df = pd.DataFrame(final_results_rows)

    first_cols = ['Sample File', 'N_Contributors', 'True Overall Proportions', 'Ensembled Predicted Proportions', 'Ensembled MAE']
    marker_cols = [col for col in final_print_df.columns if col not in first_cols]
    final_print_df = final_print_df[first_cols + sorted(marker_cols)]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(final_print_df.to_string())

    final_comparison_filename = os.path.join(MODEL_SAVE_DIR, "final_ensembled_proportions_comparison.csv")
    try:
        final_print_df.to_csv(final_comparison_filename, index=False, encoding='utf-8-sig')
        print(f"\n最终集成对比结果已保存到: {final_comparison_filename}")
        print("请确保该文件未被其他程序占用，且您有写入权限。")
    except Exception as e:
        print(f"保存最终集成对比结果时发生错误: {e}")
        print("请检查：1. 文件是否被其他程序（如Excel）占用。2. 您是否有权限在该目录下写入文件。")

    avg_ensembled_mae_list = [m for m in final_print_df['Ensembled MAE'].tolist() if isinstance(m, (float,int)) and not np.isnan(m)]
    avg_ensembled_mae = np.mean(avg_ensembled_mae_list) if avg_ensembled_mae_list else np.nan
    print(f"\n所有样本的平均集成MAE (等位基因比例): {avg_ensembled_mae:.4f}" if not np.isnan(avg_ensembled_mae) else "所有样本的平均集成MAE (等位基因比例): N/A")
    print("\n--- 所有流程执行完毕 ---")


import pandas as pd
import numpy as np
import re
import ast 
from itertools import combinations
try:
    from scipy.optimize import lsq_linear
    SCIPY_AVAILABLE = True
except ImportError:
    print("警告: 未安装 SciPy 库。比例推断将使用简化启发式方法，可能不精确。请安装 SciPy (pip install scipy)。")
    SCIPY_AVAILABLE = False

def parse_true_info_from_sample_file(sample_file_name, n_contributors):
    id_pattern = re.compile(r'RD14-0003-(\d+(?:_\d+)*)-\d+(?:;\d+)*-')
    id_match = id_pattern.search(sample_file_name)
    true_ids = []
    if id_match:
        ids_str = id_match.group(1)
        true_ids = [str(id) for id in ids_str.split('_')] 

    ratio_pattern = re.compile(r'(?:[\d_]+)-(\d+(?:;\d+)*)-[A-Za-z]')
    ratio_match = ratio_pattern.search(sample_file_name)
    true_ratios = []
    if ratio_match:
        ratio_str = ratio_match.group(1)
        try:
            parts = [float(p) for p in ratio_str.split(';')]
            total = sum(parts)
            if total > 0:
                true_ratios = [round(p / total, 4) for p in parts]
            else:
                true_ratios = [0.0] * len(parts)
        except:
            true_ratios = []

    if len(true_ids) > n_contributors:
        true_ids = true_ids[:n_contributors]
    elif len(true_ratios) > n_contributors:
        true_ratios = true_ratios[:n_contributors]
    
    if len(true_ids) < n_contributors:
        true_ids.extend(['N/A'] * (n_contributors - len(true_ids)))
    if len(true_ratios) < n_contributors:
        true_ratios.extend([0.0] * (n_contributors - len(true_ratios)))

    return true_ids, true_ratios

def load_contributor_genotypes(genotype_csv_path):
    try:
        df_genotype = pd.read_csv(genotype_csv_path)
        print(f"文件 '{genotype_csv_path}' 加载成功。")
    except FileNotFoundError:
        print(f"错误: 基因型文件 '{genotype_csv_path}' 未找到。")
        return None
    except Exception as e:
        print(f"加载基因型文件 '{genotype_csv_path}' 时发生错误: {e}")
        return None

    required_id_cols = ['Sample ID'] 
    for col in required_id_cols:
        if col not in df_genotype.columns:
            print(f"错误: 基因型文件中缺少必需的列 '{col}'。")
            return None

    contributor_genotypes = {}
    marker_cols = [col for col in df_genotype.columns if col not in ['Reseach', 'ID', 'Sample ID']]

    if not marker_cols:
        print("错误: 基因型文件中未找到任何 Marker 列。")
        return None

    for _, row in df_genotype.iterrows():
        person_id = str(row['Sample ID'])

        if person_id not in contributor_genotypes:
            contributor_genotypes[person_id] = {}
        
        for marker_col in marker_cols:
            marker_name = marker_col
            alleles_str = row.get(marker_col)

            if pd.notna(alleles_str) and isinstance(alleles_str, str):
                alleles = [a.strip() for a in alleles_str.split(',') if a.strip()]
                if len(alleles) == 2:
                    contributor_genotypes[person_id][marker_name] = sorted(alleles)
                elif len(alleles) == 1:
                    contributor_genotypes[person_id][marker_name] = sorted([alleles[0], alleles[0]])
                else:
                    contributor_genotypes[person_id][marker_name] = []
            else:
                contributor_genotypes[person_id][marker_name] = []

    print(f"加载了 {len(contributor_genotypes)} 位贡献者的基因型数据。")
    return contributor_genotypes

def infer_proportions_and_score(
    predicted_allele_proportions_by_marker, 
    combo_person_ids, 
    contributor_genotypes_db, 
    all_markers_unique, 
    allele_proportion_threshold=0.01 
):
    if not SCIPY_AVAILABLE:
        inferred_proportions = np.ones(len(combo_person_ids)) / len(combo_person_ids)
        total_rmse = 0.0
        total_alleles_compared = 0

        for marker_name, pred_props_dict in predicted_allele_proportions_by_marker.items():
            if marker_name not in all_markers_unique: continue

            theoretical_props_marker = {}
            for person_id in combo_person_ids:
                if marker_name in contributor_genotypes_db.get(person_id, {}):
                    for allele in contributor_genotypes_db[person_id][marker_name]:
                        theoretical_props_marker[str(allele)] = theoretical_props_marker.get(str(allele), 0) + (1.0 / (len(combo_person_ids) * 2))
            
            for allele_name, predicted_prop in pred_props_dict.items():
                if predicted_prop < allele_proportion_threshold: 
                    continue
                theoretical_prop = theoretical_props_marker.get(allele_name, 0.0)
                total_rmse += (predicted_prop - theoretical_prop)**2
                total_alleles_compared += 1
        
        if total_alleles_compared > 0:
            rmse_score = np.sqrt(total_rmse / total_alleles_compared)
        else:
            rmse_score = float('inf')
        
        return inferred_proportions.tolist(), rmse_score

    all_alleles_in_system = set()
    for marker in all_markers_unique:
        if marker in predicted_allele_proportions_by_marker:
            for allele_name in predicted_allele_proportions_by_marker[marker].keys():
                all_alleles_in_system.add(allele_name)
        for person_id in combo_person_ids:
            if marker in contributor_genotypes_db.get(person_id, {}):
                for allele in contributor_genotypes_db[person_id][marker]:
                    all_alleles_in_system.add(str(allele))

    sorted_alleles_in_system = sorted(list(all_alleles_in_system))
    allele_to_row_idx = {allele: i for i, allele in enumerate(sorted_alleles_in_system)}
    
    num_alleles = len(sorted_alleles_in_system)
    num_persons = len(combo_person_ids)

    if num_alleles == 0:
        return np.zeros(num_persons).tolist(), float('inf')

    A = np.zeros((num_alleles, num_persons))
    b = np.zeros(num_alleles)
    weights = np.ones(num_alleles) 
    for marker, pred_props_dict in predicted_allele_proportions_by_marker.items():
        for allele_name, predicted_prop in pred_props_dict.items():
            if allele_name in allele_to_row_idx:
                idx = allele_to_row_idx[allele_name]
                b[idx] += predicted_prop
                if predicted_prop < allele_proportion_threshold:
                    weights[idx] *= 0.1 

    total_predicted_prop_in_b = np.sum(b)
    if total_predicted_prop_in_b > 0:
        b /= total_predicted_prop_in_b
    else:
        pass

    A_weighted = A * np.sqrt(weights[:, np.newaxis])
    b_weighted = b * np.sqrt(weights)

    person_total_alleles_in_relevant_markers = np.zeros(num_persons)
    for person_idx, person_id in enumerate(combo_person_ids):
        for marker in all_markers_unique:
            if marker in contributor_genotypes_db.get(person_id, {}):
                person_total_alleles_in_relevant_markers[person_idx] += len(contributor_genotypes_db[person_id][marker])
    
    for person_idx, person_id in enumerate(combo_person_ids):
        for marker in all_markers_unique:
            if marker in contributor_genotypes_db.get(person_id, {}):
                for allele_in_genotype in contributor_genotypes_db[person_id][marker]:
                    str_allele_in_genotype = str(allele_in_genotype)
                    if str_allele_in_genotype in allele_to_row_idx:
                        if person_total_alleles_in_relevant_markers[person_idx] > 0:
                            A[allele_to_row_idx[str_allele_in_genotype], person_idx] += 1.0 / person_total_alleles_in_relevant_markers[person_idx]

    A_weighted = A * np.sqrt(weights[:, np.newaxis])
    result = lsq_linear(A_weighted, b_weighted, bounds=(0, np.inf))
    inferred_proportions_arr = result.x
    prop_sum = np.sum(inferred_proportions_arr)
    if prop_sum > 0:
        inferred_proportions_arr /= prop_sum
    else:
        inferred_proportions_arr = np.zeros(num_persons)
    
    theoretical_proportions_b = A @ inferred_proportions_arr 
    rmse_score = np.sqrt(np.sum((b - theoretical_proportions_b)**2)) 
    return inferred_proportions_arr.tolist(), rmse_score

def identify_contributors_and_proportions(
    final_predictions_df,
    contributor_genotypes_db,
    original_full_df, 
    all_markers_unique,
    max_people_for_ratio_task 
    ):
    
    identified_results = []

    all_contributor_ids = sorted(list(contributor_genotypes_db.keys()))
    print("\n--- 开始识别贡献者和推断混合比例 ---")
    for sample_idx, sample_row in final_predictions_df.iterrows(): 
        sample_file = sample_row['Sample File']
        n_contributors_true = sample_row['N_Contributors']

        predicted_allele_proportions_by_marker = {}
        
        for marker_name in all_markers_unique:
            pred_col_name = f'Pred Allele Proportions ({marker_name})'
            if pred_col_name in sample_row and isinstance(sample_row[pred_col_name], list) and len(sample_row[pred_col_name]) > 0:
                predicted_proportions_list = sample_row[pred_col_name]
                
                marker_subset_df_prog2 = original_full_df[original_full_df['Marker'] == marker_name]
                if marker_subset_df_prog2.empty: continue
                
                current_marker_canonical_alleles_prog2 = set()
                for _, sub_row in marker_subset_df_prog2.iterrows():
                    current_allele_counts_dict_prog2 = sub_row['Allele_Counts_Dict']
                    for key in current_allele_counts_dict_prog2.keys():
                        current_marker_canonical_alleles_prog2.add(str(key))
                canonical_allele_names_for_marker_prog2 = sorted(list(current_marker_canonical_alleles_prog2))
                
                predicted_allele_proportions_by_marker[marker_name] = {}
                for i, allele_name in enumerate(canonical_allele_names_for_marker_prog2):
                    if i < len(predicted_proportions_list):
                        predicted_allele_proportions_by_marker[marker_name][allele_name] = predicted_proportions_list[i]
        
        print(f"\n--- Processing Sample {sample_idx + 1}/{len(final_predictions_df)}: {sample_file} (N={n_contributors_true}) ---")
        print(f"  Predicted Allele Proportions (by Marker): {predicted_allele_proportions_by_marker}")

        best_match_score = float('inf')
        best_contributor_combo = []
        best_inferred_proportions = []

        if n_contributors_true == 0:
            identified_results.append({
                'Sample File': sample_file,
                'True N': n_contributors_true,
                'Predicted N': 0,
                'True Contributors': [],
                'Predicted Contributors': [],
                'True Proportions': [],
                'Predicted Proportions': [],
                'Match Score (RMSE)': 0.0
            })
            print(f"  样本 {sample_file} 处理完成。")
            continue

        markers_with_predictions = list(predicted_allele_proportions_by_marker.keys())
        if not markers_with_predictions:
            print(f"  警告: 样本 {sample_file} 没有有效的预测等位基因比例。跳过贡献者识别。")
            true_contributor_ids, true_proportions = parse_true_info_from_sample_file(sample_file, n_contributors_true)
            identified_results.append({
                'Sample File': sample_file,
                'True N': n_contributors_true,
                'Predicted N': n_contributors_true,
                'True Contributors': true_contributor_ids,
                'Predicted Contributors': [],
                'True Proportions': true_proportions,
                'Predicted Proportions': [],
                'Match Score (RMSE)': float('inf')
            })
            print(f"  样本 {sample_file} 处理完成。")
            continue

        relevant_contributor_ids = [
            pid for pid in all_contributor_ids 
            if any(marker in contributor_genotypes_db.get(pid, {}) for marker in markers_with_predictions)
        ]
        
        if len(relevant_contributor_ids) < n_contributors_true:
            print(f"  警告: 样本 {sample_file}: 基因型数据库中相关贡献者数量 ({len(relevant_contributor_ids)}) 少于所需贡献人数 ({n_contributors_true})。无法形成组合。")
            true_contributor_ids, true_proportions = parse_true_info_from_sample_file(sample_file, n_contributors_true)
            identified_results.append({
                'Sample File': sample_file,
                'True N': n_contributors_true,
                'Predicted N': n_contributors_true,
                'True Contributors': true_contributor_ids,
                'Predicted Contributors': [],
                'True Proportions': true_proportions,
                'Predicted Proportions': [],
                'Match Score (RMSE)': float('inf')
            })
            print(f"  样本 {sample_file} 处理完成。")
            continue
        current_combo_for_greedy = []
        remaining_candidates = list(relevant_contributor_ids)

        for _ in range(n_contributors_true): 
            best_person_for_this_step = None
            lowest_rmse_for_this_step = float('inf')

            for candidate_person_id in remaining_candidates:
                test_combo = current_combo_for_greedy + [candidate_person_id]
                inferred_proportions, rmse_score = infer_proportions_and_score(
                    predicted_allele_proportions_by_marker,
                    test_combo,
                    contributor_genotypes_db,
                    all_markers_unique
                )

                if rmse_score < lowest_rmse_for_this_step:
                    lowest_rmse_for_this_step = rmse_score
                    best_person_for_this_step = candidate_person_id
                    best_inferred_proportions_for_this_step = inferred_proportions
            
            if best_person_for_this_step is not None:
                current_combo_for_greedy.append(best_person_for_this_step)
                remaining_candidates.remove(best_person_for_this_step)
                best_match_score = lowest_rmse_for_this_step
                best_inferred_proportions = best_inferred_proportions_for_this_step
            else:
                print(f"  警告: 样本 {sample_file}: 贪婪搜索未能找到足够的贡献者。")
                break 
        best_contributor_combo = current_combo_for_greedy 

        if not best_contributor_combo or len(best_contributor_combo) != n_contributors_true:
            print(f"  警告: 样本 {sample_file} 无法通过启发式搜索找到合适的贡献者组合。将返回空预测。")
            true_contributor_ids, true_proportions = parse_true_info_from_sample_file(sample_file, n_contributors_true)
            identified_results.append({
                'Sample File': sample_file,
                'True N': n_contributors_true,
                'Predicted N': n_contributors_true,
                'True Contributors': true_contributor_ids,
                'Predicted Contributors': [],
                'True Proportions': true_proportions,
                'Predicted Proportions': [],
                'Match Score (RMSE)': float('inf')
            })
            print(f"  样本 {sample_file} 处理完成。")
            continue

        true_contributor_ids, true_proportions = parse_true_info_from_sample_file(sample_file, n_contributors_true)

        identified_results.append({
            'Sample File': sample_file,
            'True N': n_contributors_true,
            'Predicted N': n_contributors_true,
            'True Contributors': true_contributor_ids,
            'Predicted Contributors': best_contributor_combo,
            'True Proportions': true_proportions,
            'Predicted Proportions': [round(p, 4) for p in best_inferred_proportions],
            'Match Score (RMSE)': round(best_match_score, 4)
        })
        print(f"  样本 {sample_file} 处理完成。") 

    return pd.DataFrame(identified_results)

if __name__ == '__main__':
    PREDICTED_PROPORTIONS_CSV_PATH = 'trained_marker_proportion_models/final_ensembled_proportions_comparison.csv'
    GENOTYPE_CSV_PATH = '附件3：各个贡献者对应的基因型数据.csv'
    ORIGINAL_CSV_PATH_FOR_PROG2 = 'p3k1_with_prediction.csv'
    
    MODEL_SAVE_DIR = "trained_marker_proportion_models" 
    MAX_PEOPLE_FOR_RATIO_TASK_CONFIG = 6 
    print("--- 程序二：开始贡献者识别与比例推断 ---")
    try:
        final_predictions_df = pd.read_csv(PREDICTED_PROPORTIONS_CSV_PATH)
        print(f"文件 '{PREDICTED_PROPORTIONS_CSV_PATH}' 加载成功。")
    except FileNotFoundError:
        print(f"错误: 预测比例文件 '{PREDICTED_PROPORTIONS_CSV_PATH}' 未找到。请先运行程序一。")
        exit()
    except Exception as e:
        print(f"加载预测比例文件 '{PREDICTED_PROPORTIONS_CSV_PATH}' 时发生错误: {e}")
        exit()

    pred_prop_cols = [col for col in final_predictions_df.columns if col.startswith('Pred Allele Proportions (')]
    
    for col in pred_prop_cols:
        final_predictions_df[col] = final_predictions_df[col].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        )
        final_predictions_df[col] = final_predictions_df[col].apply(
            lambda x: [float(item) for item in x] if isinstance(x, list) else x
        )

    contributor_genotypes_db = load_contributor_genotypes(GENOTYPE_CSV_PATH)
    if contributor_genotypes_db is None:
        print("无法加载贡献者基因型数据，程序退出。请检查基因型文件路径和内容。")
        exit()
    
    try:
        original_full_df_prog2 = pd.read_csv(ORIGINAL_CSV_PATH_FOR_PROG2)
        original_full_df_prog2.loc[:, 'Allele_Counts_Dict'] = original_full_df_prog2['Allele_Counts_From_Contributors'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else {}
        )
        print(f"原始数据文件 '{ORIGINAL_CSV_PATH_FOR_PROG2}' 在程序二中加载成功。")
    except Exception as e:
        print(f"错误: 无法在程序二中加载原始数据文件 '{ORIGINAL_CSV_PATH_FOR_PROG2}'，用于重构规范等位基因名称: {e}")
        exit()

    all_markers_unique = sorted(original_full_df_prog2['Marker'].unique().tolist())

    final_identification_results_df = identify_contributors_and_proportions(
        final_predictions_df,
        contributor_genotypes_db,
        original_full_df_prog2,
        all_markers_unique,
        MAX_PEOPLE_FOR_RATIO_TASK_CONFIG
    )

    print("\n\n--- 最终贡献者识别与比例推断结果 ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(final_identification_results_df.to_string())

    identification_filename = os.path.join(MODEL_SAVE_DIR, "final_contributor_identification_comparison.csv")
    try:
        final_identification_results_df.to_csv(identification_filename, index=False, encoding='utf-8-sig')
        print(f"\n最终贡献者识别结果已保存到: {identification_filename}")
        print("请确保该文件未被其他程序占用，且您有写入权限。")
    except Exception as e:
        print(f"保存最终贡献者识别结果时发生错误: {e}")
        print("请检查：1. 文件是否被其他程序（如Excel）占用。2. 您是否有权限在该目录下写入文件。")
    print("\n--- 程序二：流程执行完毕 ---")
