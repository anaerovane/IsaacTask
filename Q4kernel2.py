import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.utils import class_weight 
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate, Layer, 
                                     TimeDistributed, GlobalAveragePooling1D, 
                                     BatchNormalization, Add, RepeatVector, LeakyReLU, Lambda)
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.utils import to_categorical
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns

# 根据您的要求定义的可视化配色方案
CUSTOM_PALETTE = {
    'red': '#C52A20',        # R197 G42 B32
    'black': '#000000',      # R0 G0 B0
    'blue': '#508AB2',       # R80 G138 B178
    'yellow_gold': '#F0BB41',# R240 G187 B65
    'beige': '#D5BA82',      # R213 G186 B130
    'muted_red': '#B36A6F',  # R179 G106 B111
    'mint_green': '#A1D0C7'  # R161 G208 B199
}

# --- 默认参数 ---
DEFAULT_MAX_ALLELES_PER_MARKER = 20 
MAX_CONTRIBUTORS_ASSUMED = 5 
MAX_K_CLASSES_FOR_MODEL_1 = (MAX_CONTRIBUTORS_ASSUMED * 2) + 1 


def parse_sample_file(sample_file_name): 
    mixing_ratios = []
    m_val = 0.0; ip_val = 0.0; q_val = 0.0; sec_val = 0.0; sample_prefix = ''
    parts = sample_file_name.split('-')
    true_num_people = 0
    if len(parts) > 2:
        person_ids_str = parts[2]
        if person_ids_str: true_num_people = len(person_ids_str.split('_')) 
    if len(parts) > 3:
        ratios_str = parts[3]
        try:
            raw_ratios = [int(r) for r in ratios_str.split(';')]
            total_ratio_sum = sum(raw_ratios)
            if total_ratio_sum > 0: mixing_ratios = [r / total_ratio_sum for r in raw_ratios]
            else: mixing_ratios = [0.0] * len(raw_ratios) 
        except ValueError: mixing_ratios = [1.0 / true_num_people] * true_num_people if true_num_people > 0 else []
        if len(mixing_ratios) != true_num_people: mixing_ratios = [1.0 / true_num_people] * true_num_people if true_num_people > 0 else []
        mixing_ratios.sort(reverse=True) 
    else: 
        if true_num_people > 0: mixing_ratios = [1.0 / true_num_people] * true_num_people
        else: mixing_ratios = [] 
    m_val_match = re.search(r'M(\d+\.?\d*)(?:e|S)?', sample_file_name)
    if m_val_match: m_val = float(m_val_match.group(1))
    ip_val_match = re.search(r'(\d+\.?\d*)IP', sample_file_name)
    if ip_val_match: ip_val = float(ip_val_match.group(1))
    q_val_match = re.search(r'Q(\d+\.?\d*)', sample_file_name)
    if q_val_match: q_val = float(q_val_match.group(1))
    sec_val_match = re.search(r'(\d+\.?\d*)sec', sample_file_name)
    if sec_val_match: sec_val = float(sec_val_match.group(1))
    prefix_match = re.match(r'([A-Z]\d{2}_RD\d{2}-\d{4})', sample_file_name)
    if prefix_match: sample_prefix = prefix_match.group(1)
    return true_num_people, mixing_ratios, m_val, ip_val, q_val, sec_val, sample_prefix

def melt_dataframe_robust(df, id_cols, max_alleles_per_marker, is_raw_data=True):
    melted_data_list = []
    for _, row in df.iterrows():
        base_entry = {col: row[col] for col in id_cols}
        
        for i in range(1, max_alleles_per_marker + 1):
            peak_entry = base_entry.copy()
            peak_entry['PeakNum'] = i
            size_val, allele_val, height_val = None, None, None
            
            for prefix in ['Size', 'Allele', 'Height']:
                val_no_space = row.get(f'{prefix}{i}')
                val_space = row.get(f'{prefix} {i}')
                final_val = val_no_space if pd.notna(val_no_space) else val_space

                if prefix == 'Size': size_val = final_val
                elif prefix == 'Allele': allele_val = final_val
                elif prefix == 'Height': height_val = final_val
            
            if is_raw_data:
                peak_entry['Raw_Size'], peak_entry['Raw_Allele'], peak_entry['Raw_Height'] = size_val, allele_val, height_val
            else: 
                peak_entry['Denoised_Size'], peak_entry['Denoised_Allele'], peak_entry['Denoised_Height'] = size_val, allele_val, height_val
            melted_data_list.append(peak_entry)
            
    melted_df = pd.DataFrame(melted_data_list)
    if is_raw_data:
        melted_df['Raw_Size'] = pd.to_numeric(melted_df['Raw_Size'], errors='coerce').fillna(0)
        melted_df['Raw_Height'] = pd.to_numeric(melted_df['Raw_Height'], errors='coerce').fillna(0)
        melted_df['Raw_Allele'] = melted_df['Raw_Allele'].fillna("N/A") 
    else:
        melted_df['Denoised_Size'] = pd.to_numeric(melted_df['Denoised_Size'], errors='coerce').fillna(0)
        melted_df['Denoised_Height'] = pd.to_numeric(melted_df['Denoised_Height'], errors='coerce').fillna(0)
        melted_df['Denoised_Allele'] = melted_df['Denoised_Allele'].fillna("N/A")
    return melted_df

def calculate_height_rank(series_height):
    temp_height = series_height.replace(0, -np.inf) 
    ranks = temp_height.rank(method='dense', ascending=False).astype(int)
    ranks[series_height == 0] = ranks.max() + 1 if not ranks.empty and ranks.max() > 0 else 1 
    return ranks

def preprocess_data_for_three_models(raw_file_path, denoised_file_path, default_max_alleles, max_k_classes_for_model1):
    print(f"  正在加载原始数据: {raw_file_path}")
    try: raw_df_orig = pd.read_csv(raw_file_path)
    except FileNotFoundError: print(f"错误: 原始数据文件 '{raw_file_path}' 未找到。"); return None, None, None, None
    print(f"  正在加载去噪 (真实峰) 数据 (附件4): {denoised_file_path}")
    try: denoised_df_orig = pd.read_csv(denoised_file_path) 
    except FileNotFoundError: print(f"错误: 去噪数据文件 '{denoised_file_path}' 未找到。"); return None, None, None, None

    raw_df = raw_df_orig.copy()
    denoised_df_target = denoised_df_orig.copy() 

    if 'Dye' not in raw_df.columns:
        print("警告: 附件1中未找到 'Dye' 列。将用 'Unknown' 填充。")
        raw_df['Dye'] = 'Unknown'

    height_cols_raw = [col for col in raw_df.columns if col.startswith('Height') or col.startswith('Height ')]
    if height_cols_raw: 
        nums = []
        for col in height_cols_raw:
            match = re.search(r'(\d+)', col)
            if match: nums.append(int(match.group(1)))
        if nums: max_alleles_per_marker = max(nums)
        else: max_alleles_per_marker = default_max_alleles 
    else: max_alleles_per_marker = default_max_alleles
    print(f"  MAX_ALLELES_PER_MARKER (序列长度) 设置为: {max_alleles_per_marker}")

    print("  正在解析 'Sample File' 列...")
    parsed_sample_info_df = raw_df['Sample File'].apply(lambda x: pd.Series(parse_sample_file(x)))
    parsed_sample_info_df.columns = ['parsed_true_num_people', 'parsed_true_mixing_ratios', 
                                     'parsed_M_Value', 'parsed_IP_Value', 'parsed_Q_Value', 
                                     'parsed_Sec_Value', 'parsed_Sample_Prefix']
    raw_df = pd.concat([raw_df.reset_index(drop=True), parsed_sample_info_df.reset_index(drop=True)], axis=1)
    raw_df['true_num_people'] = pd.to_numeric(raw_df['parsed_true_num_people'], errors='coerce').fillna(0).astype(int)
    print(f"  'true_num_people' 列值计数 (解析后): \n{raw_df['true_num_people'].value_counts().sort_index()}")

    print("  正在转换原始数据 (附件1) 为长表格式...")
    raw_id_cols = ['Sample File', 'Marker', 'Dye', 'true_num_people']
    long_raw_df = melt_dataframe_robust(raw_df, raw_id_cols, max_alleles_per_marker, is_raw_data=True)
    sample_level_parsed_values = raw_df[['Sample File', 'parsed_M_Value', 'parsed_IP_Value', 'parsed_Q_Value', 'parsed_Sec_Value']].drop_duplicates()
    long_raw_df = pd.merge(long_raw_df, sample_level_parsed_values, on='Sample File', how='left')

    print("  正在转换去噪目标数据 (附件4) 为长表格式...")
    denoised_id_cols = ['Sample File', 'Marker', 'Dye']
    if 'Dye' not in denoised_df_target.columns: denoised_df_target['Dye'] = 'Unknown'
    long_denoised_target_df = melt_dataframe_robust(denoised_df_target, denoised_id_cols, max_alleles_per_marker, is_raw_data=False)

    print("  正在合并原始数据和去噪目标数据...")
    merged_df_for_processing = pd.merge(long_raw_df, 
                         long_denoised_target_df[['Sample File', 'Marker', 'PeakNum', 
                                           'Denoised_Size', 'Denoised_Allele', 'Denoised_Height']],
                         on=['Sample File', 'Marker', 'PeakNum'], 
                         how='left')
    for col in ['Denoised_Size', 'Denoised_Height']: merged_df_for_processing[col] = merged_df_for_processing[col].fillna(0)
    merged_df_for_processing['Denoised_Allele'] = merged_df_for_processing['Denoised_Allele'].fillna("N/A")
    
    merged_df_for_processing['IsActuallyRealPeak_M2_M3_Target'] = (merged_df_for_processing['Denoised_Height'] > 0).astype(int) 
    merged_df_for_processing['Actual_Denoised_Height_Target_M3'] = merged_df_for_processing['Denoised_Height'] 
    merged_df_for_processing['Height_Difference_Target_M3'] = 0.0
    real_peak_mask_for_m3 = merged_df_for_processing['IsActuallyRealPeak_M2_M3_Target'] == 1
    merged_df_for_processing.loc[real_peak_mask_for_m3, 'Height_Difference_Target_M3'] = \
        merged_df_for_processing.loc[real_peak_mask_for_m3, 'Actual_Denoised_Height_Target_M3'] - merged_df_for_processing.loc[real_peak_mask_for_m3, 'Raw_Height']

    print("  正在计算每个样本-Marker的真实独特等位基因总数 (模型1目标)...")
    true_peak_counts_map = {}
    for (sample_file, marker), group in long_denoised_target_df[long_denoised_target_df['Denoised_Height'] > 0].groupby(['Sample File', 'Marker']):
        unique_real_alleles_in_group = group['Denoised_Allele'].astype(str).nunique()
        true_peak_counts_map[(sample_file, marker)] = unique_real_alleles_in_group
    merged_df_for_processing['TrueKCount_Target_M1'] = merged_df_for_processing.apply(
        lambda row: true_peak_counts_map.get((row['Sample File'], row['Marker']), 0), axis=1
    )
    
    print("  正在计算峰高排名 (模型2特征)...")
    merged_df_for_processing['Height_Rank_M2_Feature_Raw'] = merged_df_for_processing.groupby(['Sample File', 'Marker'])['Raw_Height'].transform(calculate_height_rank)
    
    print("  正在进行特征缩放 (MinMaxScaler)...")
    scaler_size = MinMaxScaler(); scaler_height_common = MinMaxScaler(); scaler_rank = MinMaxScaler()
    scaler_num_contrib = MinMaxScaler(); scaler_true_k_for_m2 = MinMaxScaler()
    
    if not merged_df_for_processing[['Raw_Size']].dropna().empty: 
        merged_df_for_processing['Scaled_Raw_Size'] = scaler_size.fit_transform(merged_df_for_processing[['Raw_Size']])
    else: merged_df_for_processing['Scaled_Raw_Size'] = 0; print("警告: Raw_Size 数据为空或全为NaN。")
    
    if not merged_df_for_processing[['Raw_Height']].dropna().empty:
        merged_df_for_processing['Scaled_Raw_Height_Common'] = scaler_height_common.fit_transform(merged_df_for_processing[['Raw_Height']]) 
    else:
        merged_df_for_processing['Scaled_Raw_Height_Common'] = 0; 
        print("警告: Raw_Height 数据为空或全为NaN，Scaled_Raw_Height_Common 将为0。")
    
    if not merged_df_for_processing[['Height_Rank_M2_Feature_Raw']].dropna().empty:
        merged_df_for_processing['Scaled_Height_Rank_M2_Feature'] = scaler_rank.fit_transform(merged_df_for_processing[['Height_Rank_M2_Feature_Raw']])
    else: merged_df_for_processing['Scaled_Height_Rank_M2_Feature'] = 0; print("警告: Height_Rank_M2_Feature_Raw 数据为空或全为NaN。")

    if not merged_df_for_processing[['true_num_people']].dropna().empty:
         merged_df_for_processing['Scaled_Num_Contributors_M2_Feature'] = scaler_num_contrib.fit_transform(merged_df_for_processing[['true_num_people']])
    else: merged_df_for_processing['Scaled_Num_Contributors_M2_Feature'] = 0; print("警告: true_num_people 数据为空或全为NaN。")

    if not merged_df_for_processing[['TrueKCount_Target_M1']].dropna().empty:
        merged_df_for_processing['Scaled_TrueK_M2_Feature'] = scaler_true_k_for_m2.fit_transform(merged_df_for_processing[['TrueKCount_Target_M1']])
    else:
        merged_df_for_processing['Scaled_TrueK_M2_Feature'] = 0; print("警告: TrueKCount_Target_M1 数据为空或全为NaN。")

    print("  正在准备模型1的辅助统计特征...")
    aux_stats_m1_list = []
    for (sample_file, marker_name_stats), group_for_stats in merged_df_for_processing.groupby(['Sample File', 'Marker']): 
        raw_heights = group_for_stats['Raw_Height'].values 
        effective_peaks_mask = raw_heights > 50 
        num_raw_peaks_above_thresh = np.sum(effective_peaks_mask)
        if num_raw_peaks_above_thresh > 0:
            mean_raw_height_above_thresh = np.mean(raw_heights[effective_peaks_mask])
            std_raw_height_above_thresh = np.std(raw_heights[effective_peaks_mask])
            max_raw_height_above_thresh = np.max(raw_heights[effective_peaks_mask])
        else: mean_raw_height_above_thresh, std_raw_height_above_thresh, max_raw_height_above_thresh = 0,0,0
        total_raw_height_all = np.sum(raw_heights)
        mean_raw_height_all = np.mean(raw_heights) if len(raw_heights) > 0 else 0
        m_val = group_for_stats['parsed_M_Value'].iloc[0] if 'parsed_M_Value' in group_for_stats.columns else 0
        ip_val = group_for_stats['parsed_IP_Value'].iloc[0] if 'parsed_IP_Value' in group_for_stats.columns else 0
        q_val = group_for_stats['parsed_Q_Value'].iloc[0] if 'parsed_Q_Value' in group_for_stats.columns else 0
        sec_val = group_for_stats['parsed_Sec_Value'].iloc[0] if 'parsed_Sec_Value' in group_for_stats.columns else 0
        aux_stats_m1_list.append({
            'Sample File': sample_file, 'Marker': marker_name_stats,
            'num_raw_peaks_above_thresh': num_raw_peaks_above_thresh,
            'mean_raw_height_above_thresh': mean_raw_height_above_thresh,
            'std_raw_height_above_thresh': std_raw_height_above_thresh,
            'max_raw_height_above_thresh': max_raw_height_above_thresh,
            'total_raw_height_all': total_raw_height_all, 'mean_raw_height_all': mean_raw_height_all,
            'm_val': m_val, 'ip_val': ip_val, 'q_val': q_val, 'sec_val': sec_val
        })
    aux_stats_m1_df_local = pd.DataFrame(aux_stats_m1_list) 
    cols_to_scale_aux_m1 = ['num_raw_peaks_above_thresh', 'mean_raw_height_above_thresh', 
                            'std_raw_height_above_thresh', 'max_raw_height_above_thresh',
                            'total_raw_height_all', 'mean_raw_height_all',
                            'm_val', 'ip_val', 'q_val', 'sec_val'] 
    scalers_aux_m1 = {col: MinMaxScaler() for col in cols_to_scale_aux_m1}
    if not aux_stats_m1_df_local.empty:
        for col in cols_to_scale_aux_m1:
            if col in aux_stats_m1_df_local.columns and not aux_stats_m1_df_local[[col]].dropna().empty:
                 aux_stats_m1_df_local[f'scaled_{col}'] = scalers_aux_m1[col].fit_transform(aux_stats_m1_df_local[[col]])
            else: aux_stats_m1_df_local[f'scaled_{col}'] = 0; print(f"警告: 辅助特征 {col} 数据为空、全为NaN或不存在于aux_stats_m1_df_local。")
    else: print("警告: aux_stats_m1_df_local 为空，无法进行辅助特征缩放。")

    data_by_marker = {}
    print("  按Marker分组数据，为三个模型准备输入...")
    for marker_name_loop, marker_group_df in merged_df_for_processing.groupby('Marker'): 
        X_peak_seq_m1_list, X_num_contrib_m1_list, X_aux_other_m1_list, y_true_k_count_m1_list = [], [], [], []
        X_peak_seq_m2_attn_list, X_peak_seq_m2_add_list, X_aux_m2_list, y_peak_likelihood_m2_list = [], [], [], []
        X_m3_true_peak_seq_list, X_m3_additional_true_peak_features_list, X_m3_sample_level_aux_list, y_m3_target_height_difference_list = [], [], [], []
        m3_sample_true_peak_counts = [] 

        sample_identifiers_list = []

        for sample_file, sample_marker_data in marker_group_df.groupby('Sample File'):
            sample_marker_data = sample_marker_data.sort_values(by='PeakNum')
            num_contrib_orig = sample_marker_data['true_num_people'].iloc[0]
            scaled_num_contrib_val_m2 = sample_marker_data['Scaled_Num_Contributors_M2_Feature'].iloc[0]
            true_overall_k_count = sample_marker_data['TrueKCount_Target_M1'].iloc[0]
            scaled_true_k_val_m2 = sample_marker_data['Scaled_TrueK_M2_Feature'].iloc[0]

            current_peak_seq_m1 = np.zeros((max_alleles_per_marker, 2)) 
            current_peak_seq_m2_attn = np.zeros((max_alleles_per_marker, 2)) 
            current_peak_seq_m2_add = np.zeros((max_alleles_per_marker, 2)) 
            current_likelihood_seq_m2 = np.zeros((max_alleles_per_marker, 1))
            
            current_sample_m3_peak_seq_attn_buffer = []
            current_sample_m3_additional_peak_features_buffer = []
            current_sample_m3_height_diff_buffer = []

            for i in range(max_alleles_per_marker):
                peak_num_to_find = i + 1
                row_for_peak = sample_marker_data[sample_marker_data['PeakNum'] == peak_num_to_find]
                if not row_for_peak.empty:
                    row_data = row_for_peak.iloc[0]
                    current_peak_seq_m1[i, 0] = row_data['Scaled_Raw_Size']
                    current_peak_seq_m1[i, 1] = row_data['Scaled_Raw_Height_Common']
                    current_peak_seq_m2_attn[i, 0] = row_data['Scaled_Raw_Size']
                    current_peak_seq_m2_attn[i, 1] = row_data['Scaled_Raw_Height_Common']
                    current_peak_seq_m2_add[i, 0] = row_data['Raw_Height'] 
                    current_peak_seq_m2_add[i, 1] = row_data['Scaled_Height_Rank_M2_Feature']
                    current_likelihood_seq_m2[i, 0] = row_data['IsActuallyRealPeak_M2_M3_Target']
                    
                    if row_data['IsActuallyRealPeak_M2_M3_Target'] == 1: 
                        current_sample_m3_peak_seq_attn_buffer.append([row_data['Scaled_Raw_Size'], row_data['Scaled_Raw_Height_Common']])
                        current_sample_m3_additional_peak_features_buffer.append([row_data['Raw_Height'], row_data['Scaled_Height_Rank_M2_Feature']])
                        current_sample_m3_height_diff_buffer.append(row_data['Height_Difference_Target_M3'])
            
            X_peak_seq_m1_list.append(current_peak_seq_m1)
            X_num_contrib_m1_list.append(num_contrib_orig) 
            y_true_k_count_m1_list.append(min(true_overall_k_count, max_k_classes_for_model1 - 1)) 
            
            current_aux_stats_m1_row = aux_stats_m1_df_local[(aux_stats_m1_df_local['Sample File'] == sample_file) & (aux_stats_m1_df_local['Marker'] == marker_name_loop)] 
            if not current_aux_stats_m1_row.empty:
                scaled_cols = [f'scaled_{col}' for col in cols_to_scale_aux_m1 if f'scaled_{col}' in current_aux_stats_m1_row]
                if len(scaled_cols) == len(cols_to_scale_aux_m1): 
                    aux_features_for_sample_m1 = current_aux_stats_m1_row[scaled_cols].iloc[0].values.tolist()
                    X_aux_other_m1_list.append(aux_features_for_sample_m1)
                else: X_aux_other_m1_list.append([0] * len(cols_to_scale_aux_m1))
            else: X_aux_other_m1_list.append([0] * len(cols_to_scale_aux_m1))

            X_peak_seq_m2_attn_list.append(current_peak_seq_m2_attn)
            X_peak_seq_m2_add_list.append(current_peak_seq_m2_add)
            X_aux_m2_list.append([scaled_num_contrib_val_m2, scaled_true_k_val_m2])
            y_peak_likelihood_m2_list.append(current_likelihood_seq_m2)
            
            if current_sample_m3_peak_seq_attn_buffer: 
                 num_true_peaks_in_sample = len(current_sample_m3_peak_seq_attn_buffer)
                 m3_sample_true_peak_counts.append(num_true_peaks_in_sample)
                 X_m3_true_peak_seq_list.append(current_sample_m3_peak_seq_attn_buffer) 
                 X_m3_additional_true_peak_features_list.append(current_sample_m3_additional_peak_features_buffer)
                 y_m3_target_height_difference_list.append(current_sample_m3_height_diff_buffer) 
                 
                 m_val_m3, ip_val_m3, q_val_m3, sec_val_m3 = 0,0,0,0
                 if not current_aux_stats_m1_row.empty: 
                    m_val_m3 = current_aux_stats_m1_row['m_val'].iloc[0]
                    ip_val_m3 = current_aux_stats_m1_row['ip_val'].iloc[0]
                    q_val_m3 = current_aux_stats_m1_row['q_val'].iloc[0]
                    sec_val_m3 = current_aux_stats_m1_row['sec_val'].iloc[0]
                 m3_sample_aux = [num_contrib_orig, true_overall_k_count, m_val_m3, ip_val_m3, q_val_m3, sec_val_m3]
                 X_m3_sample_level_aux_list.append(m3_sample_aux)

            sample_identifiers_list.append({'Sample File': sample_file, 'Marker': marker_name_loop, 
                                            'NumContributors_parsed': num_contrib_orig,
                                            'TrueK_Overall': true_overall_k_count})
        if not X_peak_seq_m1_list: continue 
        
        max_k_for_m3_in_marker = max(m3_sample_true_peak_counts) if m3_sample_true_peak_counts else 0
        num_m3_samples = len(X_m3_true_peak_seq_list) 
        
        num_peak_features_for_attn_m3 = 2 
        num_additional_peak_features_m3 = 2
        num_sample_level_aux_features_m3 = 6

        padded_X_m3_true_peak_seq = np.zeros((num_m3_samples, max_k_for_m3_in_marker, num_peak_features_for_attn_m3)) 
        padded_X_m3_additional_peak_features = np.zeros((num_m3_samples, max_k_for_m3_in_marker, num_additional_peak_features_m3)) 
        padded_y_m3_height_difference = np.zeros((num_m3_samples, max_k_for_m3_in_marker, 1))

        for idx, seq_list in enumerate(X_m3_true_peak_seq_list): 
            seq_len = len(seq_list) 
            if seq_len > 0 and seq_len <= max_k_for_m3_in_marker : 
                padded_X_m3_true_peak_seq[idx, :seq_len, :] = np.array(seq_list)
                padded_X_m3_additional_peak_features[idx, :seq_len, :] = np.array(X_m3_additional_true_peak_features_list[idx])
                padded_y_m3_height_difference[idx, :seq_len, 0] = np.array(y_m3_target_height_difference_list[idx])
        
        data_by_marker[marker_name_loop] = {
            "model1_X_peak_seq": np.array(X_peak_seq_m1_list),
            "model1_X_num_contrib": np.array(X_num_contrib_m1_list).reshape(-1, 1),
            "model1_X_aux_other": np.array(X_aux_other_m1_list), 
            "model1_y_k_counts_categorical": to_categorical(np.array(y_true_k_count_m1_list), num_classes=max_k_classes_for_model1),
            "model1_y_k_counts_raw": np.array(y_true_k_count_m1_list),
            
            "model2_X_peak_seq_for_attn": np.array(X_peak_seq_m2_attn_list),
            "model2_X_additional_peak_features": np.array(X_peak_seq_m2_add_list),
            "model2_X_sample_level_aux": np.array(X_aux_m2_list), 
            "model2_y_likelihoods": np.array(y_peak_likelihood_m2_list),

            "model3_X_true_peak_seq_padded": padded_X_m3_true_peak_seq if max_k_for_m3_in_marker > 0 else np.array([]).reshape(0,0,num_peak_features_for_attn_m3),
            "model3_X_additional_true_peak_features_padded": padded_X_m3_additional_peak_features if max_k_for_m3_in_marker > 0 else np.array([]).reshape(0,0,num_additional_peak_features_m3),
            "model3_X_sample_level_aux": np.array(X_m3_sample_level_aux_list) if X_m3_sample_level_aux_list else np.array([]).reshape(0,num_sample_level_aux_features_m3), 
            "model3_y_target_height_difference_padded": padded_y_m3_height_difference if max_k_for_m3_in_marker > 0 else np.array([]).reshape(0,0,1),
            "model3_actual_seq_lengths": np.array(m3_sample_true_peak_counts), 
            "model3_max_k_for_this_marker": max_k_for_m3_in_marker,

            "identifiers": sample_identifiers_list, 
            "max_alleles_per_marker": max_alleles_per_marker 
        }
    print(f"  数据已按 {len(data_by_marker)} 个Marker进行分组，并为三个模型准备好输入。")
    if data_by_marker and len(data_by_marker) > 0: 
        first_marker_name = list(data_by_marker.keys())[0]
        print(f"\n--- {first_marker_name} 的预处理数据样本 ---")
        marker_data_sample = data_by_marker[first_marker_name]
        num_samples_to_print = min(1, marker_data_sample["model1_X_peak_seq"].shape[0])
        if num_samples_to_print > 0:
            for i in range(num_samples_to_print):
                print(f"  样本 {i+1} ({marker_data_sample['identifiers'][i]['Sample File']}):")
                print(f"    Model 1 - X_peak_seq (前3峰): \n{marker_data_sample['model1_X_peak_seq'][i, :3, :]}")
                print(f"    Model 1 - X_num_contrib: {marker_data_sample['model1_X_num_contrib'][i, 0]}")
                print(f"    Model 1 - X_aux_other (shape): {marker_data_sample['model1_X_aux_other'][i].shape}, Values (部分): {marker_data_sample['model1_X_aux_other'][i, :5]}") 
                print(f"    Model 1 - y_k_counts_raw: {marker_data_sample['model1_y_k_counts_raw'][i]}")
        else: print(f"  Marker {first_marker_name} 没有足够的样本可供打印。")
    
    all_scalers = {"size": scaler_size, "height_common": scaler_height_common, 
                   "num_contrib": scaler_num_contrib, "rank": scaler_rank, 
                   "true_k_m2": scaler_true_k_for_m2}
    all_scalers.update(scalers_aux_m1) 

    return data_by_marker, all_scalers, merged_df_for_processing, aux_stats_m1_df_local

print("\n--- Part 2: 模型构建与解决 ---")

class DistanceWeightedSelfAttention(Layer): 
    def __init__(self, d_model, lambda_decay=0.5, **kwargs):
        super(DistanceWeightedSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model; self.lambda_decay = lambda_decay 
        self.query_dense = Dense(d_model, use_bias=False, name='query_dense')
        self.key_dense = Dense(d_model, use_bias=False, name='key_dense')
        self.value_dense = Dense(d_model, use_bias=False, name='value_dense')
    def build(self, input_shape): super(DistanceWeightedSelfAttention, self).build(input_shape)
    def call(self, inputs): 
        normalized_sizes = inputs[:, :, 0] 
        normalized_heights = inputs[:, :, 1] 
        q = self.query_dense(tf.expand_dims(normalized_heights, -1)) 
        k = self.key_dense(tf.expand_dims(normalized_heights, -1))  
        v = self.value_dense(tf.expand_dims(normalized_heights, -1))   
        matmul_qk = tf.matmul(q, k, transpose_b=True) 
        dk = tf.cast(tf.shape(k)[-1], tf.float32) 
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        size_distances = tf.abs(tf.expand_dims(normalized_sizes, -1) - tf.expand_dims(normalized_sizes, -2))
        distance_penalty = -self.lambda_decay * size_distances
        modified_attention_logits = scaled_attention_logits + distance_penalty
        attention_mask = tf.cast(tf.not_equal(normalized_heights, 0), tf.float32) 
        key_mask = tf.expand_dims(attention_mask, axis=1)    
        query_mask = tf.expand_dims(attention_mask, axis=-1) 
        modified_attention_logits = modified_attention_logits + (1.0 - key_mask) * -1e9
        modified_attention_logits = modified_attention_logits + (1.0 - query_mask) * -1e9 
        attention_weights = tf.nn.softmax(modified_attention_logits, axis=-1)
        attention_weights = attention_weights * key_mask 
        output = tf.matmul(attention_weights, v) 
        return output 
    def get_config(self):
        config = super(DistanceWeightedSelfAttention, self).get_config()
        config.update({"d_model": self.d_model, "lambda_decay": self.lambda_decay})
        return config

def create_k_prediction_model_optimized(sequence_length, num_features_in_peak_seq, 
                                        num_contrib_aux_features, 
                                        num_other_aux_features, 
                                        d_model_attention, lambda_decay_attention, 
                                        max_k_classes, initial_learning_rate=0.0003, l2_reg_val=0.00001,
                                        alpha_combine=0.15): 
    peak_input_m1 = Input(shape=(sequence_length, num_features_in_peak_seq), name='peak_input_m1')
    true_num_people_input_m1 = Input(shape=(num_contrib_aux_features,), name='true_num_people_input_m1') 
    other_aux_input_m1 = Input(shape=(num_other_aux_features,), name='other_aux_input_m1') 
    strong_attention_output_m1 = DistanceWeightedSelfAttention(d_model=d_model_attention, 
                                                     lambda_decay=lambda_decay_attention,
                                                     name='strong_attention_m1')(peak_input_m1)
    weak_attention_output_m1 = DistanceWeightedSelfAttention(d_model=d_model_attention, 
                                                     lambda_decay=0.0,
                                                     name='weak_attention_m1')(peak_input_m1)
 
    scaled_weak_attention_m1 = Lambda(lambda x: x * alpha_combine, name='scale_weak_attention_m1')(weak_attention_output_m1)
    attention_output_m1 = Add(name='combine_attention_m1')([strong_attention_output_m1, scaled_weak_attention_m1])

    peak_features_agg = GlobalAveragePooling1D(name='peak_features_agg_m1')(attention_output_m1)
    pf = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(peak_features_agg)
    pf = LeakyReLU(negative_slope=0.1)(pf); pf = BatchNormalization()(pf) 
    pf = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(pf)
    pf = LeakyReLU(negative_slope=0.1)(pf); pf = BatchNormalization()(pf)
    pf = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(pf)
    pf = LeakyReLU(negative_slope=0.1)(pf); peak_features_processed = BatchNormalization()(pf)
    np_aux = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(true_num_people_input_m1)
    np_aux = LeakyReLU(negative_slope=0.1)(np_aux); np_aux = BatchNormalization()(np_aux)
    np_aux = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(np_aux)
    np_aux = LeakyReLU(negative_slope=0.1)(np_aux); num_people_processed = BatchNormalization()(np_aux)
    oa_aux = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(other_aux_input_m1)
    oa_aux = LeakyReLU(negative_slope=0.1)(oa_aux); oa_aux = BatchNormalization()(oa_aux)
    oa_aux = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(oa_aux)
    oa_aux = LeakyReLU(negative_slope=0.1)(oa_aux); other_aux_processed = BatchNormalization()(oa_aux)
    merged_features_m1 = Concatenate(name='concat_m1')([peak_features_processed, num_people_processed, other_aux_processed])
    x_m1 = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(merged_features_m1) 
    x_m1 = LeakyReLU(negative_slope=0.1)(x_m1); x_m1 = BatchNormalization()(x_m1)
    x_m1 = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(x_m1) 
    x_m1 = LeakyReLU(negative_slope=0.1)(x_m1); x_m1 = BatchNormalization()(x_m1)
    x_m1 = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val))(x_m1) 
    x_m1 = LeakyReLU(negative_slope=0.1)(x_m1); x_m1 = BatchNormalization()(x_m1)
    k_count_output_m1 = Dense(max_k_classes, activation='softmax', name='k_count_output_m1')(x_m1)
    model1 = Model(inputs=[peak_input_m1, true_num_people_input_m1, other_aux_input_m1], 
                   outputs=k_count_output_m1, name="K_Prediction_Model_Like_K42")
    optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
    model1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model1

def create_peak_identification_model(sequence_length, 
                                     num_peak_features_for_attn, 
                                     num_additional_peak_features, 
                                     num_sample_level_aux_features, 
                                     d_model_attention, lambda_decay_attention,
                                     initial_learning_rate=0.0005):
    peak_seq_for_attn_input = Input(shape=(sequence_length, num_peak_features_for_attn), name='peak_seq_for_attn_input_m2')
    additional_peak_features_input = Input(shape=(sequence_length, num_additional_peak_features), name='additional_peak_features_input_m2')
    sample_level_aux_input_m2 = Input(shape=(num_sample_level_aux_features,), name='sample_level_aux_input_m2') 
    attention_output_m2 = DistanceWeightedSelfAttention(d_model=d_model_attention,
                                                     lambda_decay=lambda_decay_attention,
                                                     name='attention_m2')(peak_seq_for_attn_input)
    scaled_num_contrib_feature = tf.keras.layers.Lambda(lambda x: x[:, 0:1], name='extract_scaled_num_contrib_m2')(sample_level_aux_input_m2)
    scaled_num_contrib_repeated = RepeatVector(sequence_length, name='repeat_num_contrib_m2')(scaled_num_contrib_feature)
    scaled_true_k_feature = tf.keras.layers.Lambda(lambda x: x[:, 1:2], name='extract_scaled_true_k_m2')(sample_level_aux_input_m2)
    scaled_true_k_repeated = RepeatVector(sequence_length, name='repeat_true_k_m2')(scaled_true_k_feature)
    merged_features_for_td_mlp = Concatenate(axis=-1, name='concat_for_td_mlp_m2')([
        attention_output_m2, additional_peak_features_input, 
        scaled_num_contrib_repeated, scaled_true_k_repeated
    ])
    peak_likelihood_output_m2 = TimeDistributed(Sequential([ 
        Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0001)), LeakyReLU(negative_slope=0.01), BatchNormalization(),
        Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0001)), LeakyReLU(negative_slope=0.01), BatchNormalization(),
        Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001)), LeakyReLU(negative_slope=0.01), BatchNormalization(),
        Dense(1, activation='sigmoid')
    ]), name='peak_likelihood_output_m2')(merged_features_for_td_mlp)
    model2 = Model(inputs=[peak_seq_for_attn_input, additional_peak_features_input, sample_level_aux_input_m2], 
                   outputs=peak_likelihood_output_m2, 
                   name="Peak_Identification_Model_Like_K42")
    optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
    model2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model2

def create_height_correction_model_attention(
    max_k_for_marker, 
    num_peak_features_for_attn_m3, 
    num_additional_peak_features_m3, 
    num_sample_level_aux_features_m3, 
    d_model_attention_m3, lambda_decay_attention_m3,
    initial_learning_rate=0.001, l2_reg_val=0.0001,
    alpha_combine=0.15
):
    true_peak_sequence_input_m3 = Input(shape=(max_k_for_marker, num_peak_features_for_attn_m3), name='true_peak_sequence_input_m3')
    additional_true_peak_features_input_m3 = Input(shape=(max_k_for_marker, num_additional_peak_features_m3), name='additional_true_peak_features_input_m3')
    sample_level_aux_input_m3 = Input(shape=(num_sample_level_aux_features_m3,), name='sample_level_aux_input_m3')

    strong_attention_output_m3 = DistanceWeightedSelfAttention(
        d_model=d_model_attention_m3,
        lambda_decay=lambda_decay_attention_m3,
        name='strong_attention_m3'
    )(true_peak_sequence_input_m3)
 
    weak_attention_output_m3 = DistanceWeightedSelfAttention(
        d_model=d_model_attention_m3,
        lambda_decay=0.0,
        name='weak_attention_m3'
    )(true_peak_sequence_input_m3)

    scaled_weak_attention_m3 = Lambda(lambda x: x * alpha_combine, name='scale_weak_attention_m3')(weak_attention_output_m3)
    attention_output_m3 = Add(name='combine_attention_m3')([strong_attention_output_m3, scaled_weak_attention_m3])
    
    repeated_aux_features_m3 = []
    for i in range(num_sample_level_aux_features_m3):
        aux_feat_slice = tf.keras.layers.Lambda(lambda x: x[:, i:i+1], name=f'slice_aux_m3_{i}')(sample_level_aux_input_m3)
        repeated_aux_features_m3.append(RepeatVector(max_k_for_marker, name=f'repeat_aux_m3_{i}')(aux_feat_slice))
    
    merged_features_for_td_mlp_m3 = Concatenate(axis=-1, name='concat_for_td_mlp_m3')([
        attention_output_m3, 
        additional_true_peak_features_input_m3
    ] + repeated_aux_features_m3) 
    
    height_diff_output_m3 = TimeDistributed(Sequential([
        Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val)), LeakyReLU(negative_slope=0.1), BatchNormalization(),
        Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val)), LeakyReLU(negative_slope=0.1), BatchNormalization(),
        Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_val)), LeakyReLU(negative_slope=0.1), BatchNormalization(),
        Dense(1, activation='linear') 
    ]), name='height_difference_output_m3')(merged_features_for_td_mlp_m3)

    model3 = Model(inputs=[true_peak_sequence_input_m3, additional_true_peak_features_input_m3, sample_level_aux_input_m3], 
                   outputs=height_diff_output_m3, 
                   name="Height_Correction_Model_Attention")
    optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
    model3.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model3


def train_evaluate_all_markers_three_stages(all_marker_data, global_params, global_scalers_dict, initial_merged_df, initial_aux_stats_m1_df): 
    model1_overall_results = {} 
    model3_trained_models = {} 
    final_output_data_list = [] 
    
    # 存储所有marker的评估结果，用于后续的聚合分析和可视化
    all_markers_evaluation_results = {}

    for marker_name, data_for_marker in all_marker_data.items():
        print(f"\n\n--- Processing Marker: {marker_name} ---")
        print(f"--- (开始为 Marker: {marker_name} 初始化和训练模型) ---")
        
        m1_X_peak_seq = data_for_marker["model1_X_peak_seq"]
        m1_X_num_contrib = data_for_marker["model1_X_num_contrib"] 
        m1_X_aux_other = data_for_marker["model1_X_aux_other"] 
        model1_target_y_categorical = data_for_marker["model1_y_k_counts_categorical"] 
        model1_target_y_raw = data_for_marker["model1_y_k_counts_raw"]         
        
        m2_X_peak_seq_for_attn = data_for_marker["model2_X_peak_seq_for_attn"]
        m2_X_additional_peak_features = data_for_marker["model2_X_additional_peak_features"]
        m2_X_sample_level_aux = data_for_marker["model2_X_sample_level_aux"] 
        m2_y_likelihoods = data_for_marker["model2_y_likelihoods"]

        m3_X_true_peak_seq_padded = data_for_marker.get("model3_X_true_peak_seq_padded") 
        m3_X_additional_true_peak_features_padded = data_for_marker.get("model3_X_additional_true_peak_features_padded") 
        m3_X_sample_level_aux_for_training = data_for_marker.get("model3_X_sample_level_aux") 
        m3_y_target_height_difference_padded = data_for_marker.get("model3_y_target_height_difference_padded")
        m3_actual_seq_lengths = data_for_marker.get("model3_actual_seq_lengths")
        m3_max_k_for_this_marker = data_for_marker.get("model3_max_k_for_this_marker", 0)


        sequence_length = data_for_marker["max_alleles_per_marker"]
        identifiers = data_for_marker["identifiers"] 

        if m1_X_peak_seq.shape[0] < 5: 
            print(f"  Marker {marker_name} 数据量过少 ({m1_X_peak_seq.shape[0]}个样本)，跳过。")
            continue

        print(f"\n  --- Training Model 1 (K-value Prediction) for Marker: {marker_name} ---")
        model1_fit_class_weights = None 
        if len(np.unique(model1_target_y_raw)) > 1: 
            m1_k_class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(model1_target_y_raw), y=model1_target_y_raw)
            model1_fit_class_weights = dict(zip(np.unique(model1_target_y_raw), m1_k_class_weights_values)) 
            print(f"    Model 1 - K-value Class Weights: {model1_fit_class_weights}")
        else: print(f"    Model 1 - K-value has only one class, no class weights applied for K prediction.")
        model1 = create_k_prediction_model_optimized( 
            sequence_length=sequence_length, num_features_in_peak_seq=m1_X_peak_seq.shape[2], 
            num_contrib_aux_features=m1_X_num_contrib.shape[1], num_other_aux_features=m1_X_aux_other.shape[1], 
            d_model_attention=global_params["d_model_m1"], lambda_decay_attention=global_params["lambda_decay_m1"],
            max_k_classes=MAX_K_CLASSES_FOR_MODEL_1, initial_learning_rate=global_params["lr_m1"],
            l2_reg_val=global_params["l2_reg_m1"], alpha_combine=global_params["alpha_combine_m1"]
        )
        # 修正: 将EarlyStopping的monitor改为'loss'
        early_stopping_m1 = EarlyStopping(monitor='loss', patience=global_params["patience_m1"], restore_best_weights=True, verbose=1, min_delta=1e-5)
        reduce_lr_m1 = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=global_params["patience_lr_m1"], min_lr=1e-7, verbose=1)
        model1_fit_params_dict = {'epochs': global_params["epochs_m1"], 'batch_size': global_params["batch_size_m1"], 'callbacks': [early_stopping_m1, reduce_lr_m1], 'verbose': 1}
        if model1_fit_class_weights: model1_fit_params_dict['class_weight'] = model1_fit_class_weights
        model1.fit([m1_X_peak_seq, m1_X_num_contrib, m1_X_aux_other], model1_target_y_categorical, **model1_fit_params_dict) 
        
        print(f"  Model 1 for {marker_name} - 评估 (在全数据上):")
        pred_k_probs_m1 = model1.predict([m1_X_peak_seq, m1_X_num_contrib, m1_X_aux_other])
        pred_k_values_m1_for_marker = np.argmax(pred_k_probs_m1, axis=1) 
        report_k_m1 = classification_report(model1_target_y_raw, pred_k_values_m1_for_marker, zero_division=0, digits=4, output_dict=True) 
        print("    K-value Prediction Classification Report (在全数据上):"); print(classification_report(model1_target_y_raw, pred_k_values_m1_for_marker, zero_division=0, digits=4)) 
        
        print(f"\n  --- Training Model 2 (Peak Identification) for Marker: {marker_name} ---")
        y_likelihoods_flat_for_weights_m2 = m2_y_likelihoods.reshape(-1).astype(int)
        unique_classes_likelihood_m2 = np.unique(y_likelihoods_flat_for_weights_m2)
        sample_weights_for_fit_list_m2 = None 
        if len(unique_classes_likelihood_m2) > 1:
            class_weights_values_m2 = class_weight.compute_class_weight('balanced', classes=unique_classes_likelihood_m2, y=y_likelihoods_flat_for_weights_m2)
            class_weights_map_m2 = dict(zip(unique_classes_likelihood_m2, class_weights_values_m2))
            sw_likelihood = np.vectorize(class_weights_map_m2.get)(m2_y_likelihoods.astype(int))
            sample_weights_for_fit_list_m2 = [sw_likelihood] 
        model2 = create_peak_identification_model(
            sequence_length=sequence_length,
            num_peak_features_for_attn=m2_X_peak_seq_for_attn.shape[2],
            num_additional_peak_features=m2_X_additional_peak_features.shape[2],
            num_sample_level_aux_features=m2_X_sample_level_aux.shape[1], 
            d_model_attention=global_params["d_model_m2"],
            lambda_decay_attention=global_params["lambda_decay_m2"],
            initial_learning_rate=global_params["lr_m2"]
        )
        # 修正: 将EarlyStopping的monitor改为'loss'
        early_stopping_m2 = EarlyStopping(monitor='loss', patience=global_params["patience_m2"], restore_best_weights=True, verbose=1, min_delta=1e-5)
        reduce_lr_m2 = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=global_params["patience_lr_m2"], min_lr=1e-7, verbose=1)
        model2.fit(
            [m2_X_peak_seq_for_attn, m2_X_additional_peak_features, m2_X_sample_level_aux], 
            m2_y_likelihoods,                                 
            sample_weight=sample_weights_for_fit_list_m2[0] if sample_weights_for_fit_list_m2 else None, 
            epochs=global_params["epochs_m2"], batch_size=global_params["batch_size_m2"],
            callbacks=[early_stopping_m2, reduce_lr_m2], verbose=1
        )
        print(f"  Model 2 for {marker_name} - 评估 (逐槽位似然性，在全数据上):")
        pred_likelihoods_m2_for_marker = model2.predict([m2_X_peak_seq_for_attn, m2_X_additional_peak_features, m2_X_sample_level_aux])

        print(f"\n  --- Training Model 3 (Height Difference Prediction) for Marker: {marker_name} ---")
        model3 = None 
        mae_m3, r2_m3 = -1, -1 # 初始化评估指标
        if m3_X_true_peak_seq_padded is not None and m3_X_true_peak_seq_padded.shape[0] > 5 and m3_max_k_for_this_marker > 0: 
            model3 = create_height_correction_model_attention(
                max_k_for_marker=m3_max_k_for_this_marker, 
                num_peak_features_for_attn_m3=m3_X_true_peak_seq_padded.shape[2],
                num_additional_peak_features_m3=m3_X_additional_true_peak_features_padded.shape[2],
                num_sample_level_aux_features_m3=m3_X_sample_level_aux_for_training.shape[1],
                d_model_attention_m3=global_params["d_model_m3"],
                lambda_decay_attention_m3=global_params["lambda_decay_m3"],
                initial_learning_rate=global_params["lr_m3"],
                l2_reg_val=global_params["l2_reg_m3"],
                alpha_combine=global_params["alpha_combine_m3"]
            )
            # 修正: 将EarlyStopping的monitor改为'loss'
            early_stopping_m3 = EarlyStopping(monitor='loss', patience=global_params["patience_m3"], restore_best_weights=True, verbose=1, min_delta=1e-4) 
            reduce_lr_m3 = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=global_params["patience_lr_m3"], min_lr=1e-7, verbose=1)
            m3_sample_weights_for_loss = (m3_y_target_height_difference_padded != 0).astype(float)
            model3.fit(
                [m3_X_true_peak_seq_padded, m3_X_additional_true_peak_features_padded, m3_X_sample_level_aux_for_training], 
                m3_y_target_height_difference_padded,
                sample_weight=m3_sample_weights_for_loss, 
                epochs=global_params["epochs_m3"], batch_size=global_params["batch_size_m3"],
                callbacks=[early_stopping_m3, reduce_lr_m3], verbose=1
            )
            print(f"  Model 3 for {marker_name} - 评估:")
            pred_height_diffs_m3 = model3.predict([m3_X_true_peak_seq_padded, m3_X_additional_true_peak_features_padded, m3_X_sample_level_aux_for_training])
            true_diffs_flat_m3, pred_diffs_flat_m3 = [], []
            for sample_idx_m3 in range(m3_actual_seq_lengths.shape[0]):
                actual_len = m3_actual_seq_lengths[sample_idx_m3]
                if actual_len > 0: 
                    true_diffs_flat_m3.extend(m3_y_target_height_difference_padded[sample_idx_m3, :actual_len, 0].flatten())
                    pred_diffs_flat_m3.extend(pred_height_diffs_m3[sample_idx_m3, :actual_len, 0].flatten())
            if true_diffs_flat_m3: 
                # 修正: 此处调用 mean_absolute_error 和 r2_score
                mae_m3 = mean_absolute_error(true_diffs_flat_m3, pred_diffs_flat_m3)
                r2_m3 = r2_score(true_diffs_flat_m3, pred_diffs_flat_m3)
                print(f"    MAE: {mae_m3:.4f}, R2 Score (on true peaks): {r2_m3:.4f}")
            model3_trained_models[marker_name] = model3 
        else:
            print(f"    Marker {marker_name} 没有足够的真实峰数据用于训练模型3。")
            model3_trained_models[marker_name] = None 
        
        # 将当前marker的评估结果存入字典
        all_markers_evaluation_results[marker_name] = {
            'model1_report': report_k_m1,
            'model1_preds': pred_k_values_m1_for_marker,
            'model1_true': model1_target_y_raw,
            'model2_preds_prob': pred_likelihoods_m2_for_marker,
            'model2_true': m2_y_likelihoods,
            'model3_mae': mae_m3,
            'model3_r2': r2_m3,
        }

        # --- 生成最终输出文件的逻辑 (保持不变) ---
        for i_sample in range(len(identifiers)): 
            ident = identifiers[i_sample] 
            predicted_k_from_model1 = pred_k_values_m1_for_marker[i_sample] 
            sample_peak_probs_m2 = pred_likelihoods_m2_for_marker[i_sample, :, 0] 
            
            final_selected_indices_by_model1k = []
            if predicted_k_from_model1 > 0:
                num_peaks_to_select_final = min(predicted_k_from_model1, sequence_length)
                final_selected_indices_by_model1k = np.argsort(sample_peak_probs_m2)[-num_peaks_to_select_final:]

            original_sample_marker_data_for_output = initial_merged_df[ 
                (initial_merged_df['Sample File'] == ident['Sample File']) &
                (initial_merged_df['Marker'] == marker_name)
            ].set_index('PeakNum')
            
            aux_stats_row_for_m3_pred = initial_aux_stats_m1_df[ 
                (initial_aux_stats_m1_df['Sample File'] == ident['Sample File']) & 
                (initial_aux_stats_m1_df['Marker'] == marker_name)
            ]
            m_val_p, ip_val_p, q_val_p, sec_val_p = 0,0,0,0
            if not aux_stats_row_for_m3_pred.empty:
                m_val_p = aux_stats_row_for_m3_pred['m_val'].iloc[0]; ip_val_p = aux_stats_row_for_m3_pred['ip_val'].iloc[0]
                q_val_p = aux_stats_row_for_m3_pred['q_val'].iloc[0]; sec_val_p = aux_stats_row_for_m3_pred['sec_val'].iloc[0]
            sample_level_aux_for_m3_pred_inference = np.array([[ident['NumContributors_parsed'], predicted_k_from_model1, m_val_p,ip_val_p,q_val_p,sec_val_p ]])

            current_sample_m3_input_seq_inference = []
            current_sample_m3_input_add_feats_inference = []
            original_indices_for_m3_input_inference = [] 

            if model3_trained_models.get(marker_name) is not None and len(final_selected_indices_by_model1k) > 0:
                for peak_original_idx in final_selected_indices_by_model1k:
                    peak_num_1based = peak_original_idx + 1
                    if peak_num_1based in original_sample_marker_data_for_output.index:
                        original_peak_data = original_sample_marker_data_for_output.loc[peak_num_1based]
                        if isinstance(original_peak_data, pd.DataFrame): original_peak_data = original_peak_data.iloc[0]
                        current_sample_m3_input_seq_inference.append([original_peak_data['Scaled_Raw_Size'], original_peak_data['Scaled_Raw_Height_Common']])
                        current_sample_m3_input_add_feats_inference.append([original_peak_data['Raw_Height'], original_peak_data['Scaled_Height_Rank_M2_Feature']])
                        original_indices_for_m3_input_inference.append(peak_original_idx)

            predicted_height_diffs_for_sample_inference = {} 
            if current_sample_m3_input_seq_inference: 
                num_selected_peaks_inference = len(current_sample_m3_input_seq_inference)
                trained_model3_instance = model3_trained_models.get(marker_name)
                if trained_model3_instance:
                    max_k_for_m3_marker_inference = trained_model3_instance.input_shape[0][1]
                    padded_m3_input_seq_infer = np.zeros((1, max_k_for_m3_marker_inference, 2)) 
                    padded_m3_input_add_feats_infer = np.zeros((1, max_k_for_m3_marker_inference, 2)) 
                    fill_len = min(num_selected_peaks_inference, max_k_for_m3_marker_inference)
                    padded_m3_input_seq_infer[0, :fill_len, :] = np.array(current_sample_m3_input_seq_inference)[:fill_len]
                    padded_m3_input_add_feats_infer[0, :fill_len, :] = np.array(current_sample_m3_input_add_feats_inference)[:fill_len]
                    pred_diffs_seq_infer = trained_model3_instance.predict([padded_m3_input_seq_infer, padded_m3_input_add_feats_infer, sample_level_aux_for_m3_pred_inference], verbose=0)
                    for k_idx, original_peak_idx in enumerate(original_indices_for_m3_input_inference): 
                         if k_idx < fill_len: predicted_height_diffs_for_sample_inference[original_peak_idx] = pred_diffs_seq_infer[0, k_idx, 0]

            for peak_slot_idx_1based in range(1, sequence_length + 1): 
                peak_slot_idx_0based = peak_slot_idx_1based - 1
                is_selected_by_pipeline = (peak_slot_idx_0based in final_selected_indices_by_model1k)
                final_h, final_s, final_a = 0.0, 0.0, "0"
                original_peak_data_series = original_sample_marker_data_for_output.loc[peak_slot_idx_1based] if peak_slot_idx_1based in original_sample_marker_data_for_output.index else None
                if isinstance(original_peak_data_series, pd.DataFrame): original_peak_data_series = original_peak_data_series.iloc[0] if not original_peak_data_series.empty else None
                if is_selected_by_pipeline and original_peak_data_series is not None:
                    final_s = original_peak_data_series['Raw_Size']
                    final_a = str(original_peak_data_series['Raw_Allele']) if pd.notna(original_peak_data_series['Raw_Allele']) else "0"
                    raw_h = original_peak_data_series['Raw_Height']
                    pred_diff = predicted_height_diffs_for_sample_inference.get(peak_slot_idx_0based, 0) 
                    final_h = max(0, raw_h + pred_diff)
                
                final_output_data_list.append({ 
                    'Sample File': ident['Sample File'], 'Marker': marker_name, 'PeakNum': peak_slot_idx_1based, 
                    'Size_final': final_s, 'Allele_final': final_a, 'Height_final': final_h,
                    'Actual_K_Overall': ident['TrueK_Overall'], 'Predicted_K_Model1': predicted_k_from_model1,
                    'Actual_IsRealPeak_Slot': int(m2_y_likelihoods[i_sample, peak_slot_idx_0based, 0]),
                    'Predicted_Likelihood_Model2': sample_peak_probs_m2[peak_slot_idx_0based],
                    'Is_Selected_by_Pipeline': int(is_selected_by_pipeline),
                    'Actual_Denoised_Height_from_Att4': original_peak_data_series['Actual_Denoised_Height_Target_M3'] if original_peak_data_series is not None else 0
                })
        print(f"--- Marker: {marker_name} 处理完毕 ---")

    if not final_output_data_list: print("警告：没有生成任何预测结果。"); return None, None
    final_output_df_long = pd.DataFrame(final_output_data_list) 
    return final_output_df_long, all_markers_evaluation_results


def format_output_like_附件4(df_long_results, raw_data_filepath, max_alleles_per_marker):
    print("\n--- 正在基于附件1的格式生成修改后的输出文件 ---")
    if df_long_results.empty:
        print("警告: 用于格式化的 DataFrame (df_long_results) 为空。")
        return pd.DataFrame()
    try:
        original_raw_df = pd.read_csv(raw_data_filepath)
    except FileNotFoundError:
        print(f"错误: 原始数据文件 '{raw_data_filepath}' 在格式化输出时未找到。")
        return pd.DataFrame()

    output_df = original_raw_df.copy()
    predictions_lookup = {}
    for _, row in df_long_results.iterrows():
        key = (row['Sample File'], row['Marker'])
        if key not in predictions_lookup: predictions_lookup[key] = {}
        predictions_lookup[key][row['PeakNum']] = {'selected': row['Is_Selected_by_Pipeline'], 'height': row['Height_final']}

    for index, df_row in output_df.iterrows():
        key = (df_row['Sample File'], df_row['Marker'])
        for i in range(1, max_alleles_per_marker + 1):
            height_col = f'Height{i}' if f'Height{i}' in output_df.columns else f'Height {i}'
            if height_col in output_df.columns:
                if key in predictions_lookup and i in predictions_lookup[key] and predictions_lookup[key][i]['selected'] == 1:
                    output_df.at[index, height_col] = predictions_lookup[key][i]['height']
                else:
                    output_df.at[index, height_col] = 0.0
    return output_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.' 
    raw_data_filepath = os.path.join(script_dir, "附件1：不同人数的STR图谱数据.csv")
    denoised_data_filepath = os.path.join(script_dir, "附件4：去噪后的STR图谱数据.csv")

    if not os.path.exists(raw_data_filepath): print(f"错误: 原始数据文件未找到于 '{raw_data_filepath}'"); exit()
    if not os.path.exists(denoised_data_filepath): print(f"错误: 去噪数据文件未找到于 '{denoised_data_filepath}'"); exit()

    processed_data_by_marker, global_scalers, initial_merged_df, aux_stats_m1_df_global = preprocess_data_for_three_models( 
        raw_data_filepath, 
        denoised_data_filepath,
        default_max_alleles=DEFAULT_MAX_ALLELES_PER_MARKER,
        max_k_classes_for_model1=MAX_K_CLASSES_FOR_MODEL_1
    )

    if processed_data_by_marker:
        global_training_params = {
            "d_model_m1": 192, "lambda_decay_m1": 0.005, "lr_m1": 0.0001, 
            "epochs_m1": 400, "batch_size_m1": 4, "patience_m1": 100, "patience_lr_m1": 40, 
            "l2_reg_m1": 0.00001, "alpha_combine_m1": 0.15,

            "d_model_m2": 128, "lambda_decay_m2": 0.05, "lr_m2": 0.0003,
            "epochs_m2": 400, "batch_size_m2": 8, "patience_m2": 70, "patience_lr_m2": 30, 
            
            "d_model_m3": 128, "lambda_decay_m3": 0.01, "lr_m3": 0.0005, 
            "epochs_m3": 400, "batch_size_m3": 16, 
            "patience_m3": 60, "patience_lr_m3": 25, "l2_reg_m3": 0.00005,
            "alpha_combine_m3": 0.15
        }
        final_long_df_results, all_eval_results = train_evaluate_all_markers_three_stages( 
            processed_data_by_marker,
            global_params=global_training_params,
            global_scalers_dict=global_scalers, 
            initial_merged_df=initial_merged_df, 
            initial_aux_stats_m1_df=aux_stats_m1_df_global 
        )
        
        if final_long_df_results is not None and not final_long_df_results.empty:
            print("\n\n--- 三阶段模型长格式预测结果汇总 (前50行) ---")
            print(final_long_df_results.head(50)) 
            
            # --- 新增：全面的评估与可视化部分 ---
            print("\n\n" + "="*25 + " 全面评估与可视化 " + "="*25)
            
            # 1. 整体去噪效果评估 (基于最终选择的峰)
            print("\n--- 系统级去噪性能评估 (基于最终输出) ---")
            y_true_denoise = (final_long_df_results['Actual_Denoised_Height_from_Att4'] > 0).astype(int)
            y_pred_denoise = final_long_df_results['Is_Selected_by_Pipeline']
            denoising_report = classification_report(y_true_denoise, y_pred_denoise, target_names=['噪音/伪迹', '真实峰'], output_dict=True)
            print(f"整体精确率 (真实峰): {denoising_report['真实峰']['precision']:.4f}")
            print(f"整体召回率 (真实峰): {denoising_report['真实峰']['recall']:.4f}")
            print(f"整体F1分数 (真实峰):  {denoising_report['真实峰']['f1-score']:.4f}")

            # 2. 准备绘图数据
            markers = list(all_eval_results.keys())
            model1_f1 = [all_eval_results[m]['model1_report']['macro avg']['f1-score'] for m in markers]
            model3_r2 = [all_eval_results[m]['model3_r2'] for m in markers if all_eval_results[m]['model3_r2'] != -1]
            
            # 3. 开始绘图
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体，以防图例乱码
            plt.rcParams['axes.unicode_minus'] = False

            # 图 1: 模型一 K值预测性能 (F1分数)
            plt.figure(figsize=(16, 7))
            sns.barplot(x=markers, y=model1_f1, color=CUSTOM_PALETTE['blue'], edgecolor=CUSTOM_PALETTE['black'])
            plt.title('模型一性能: 各基因座的K值预测宏平均F1分数', fontsize=18, pad=20)
            plt.ylabel('宏平均 F1-Score', fontsize=14)
            plt.xlabel('STR 基因座', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("model1_k_value_f1_scores.png")
            print("\n已保存图表: model1_k_value_f1_scores.png")
            plt.show()

            # 图 2: 模型三 峰高校正性能 (R2分数)
            plt.figure(figsize=(16, 7))
            markers_with_m3_results = [m for m in markers if all_eval_results[m]['model3_r2'] != -1]
            sns.barplot(x=markers_with_m3_results, y=model3_r2, color=CUSTOM_PALETTE['red'], edgecolor=CUSTOM_PALETTE['black'])
            plt.title('模型三性能: 各基因座的峰高校正R²分数', fontsize=18, pad=20)
            plt.ylabel('R² Score (决定系数)', fontsize=14)
            plt.xlabel('STR 基因座', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("model3_height_correction_r2_scores.png")
            print("已保存图表: model3_height_correction_r2_scores.png")
            plt.show()

            # 图 3: 去噪效果示例对比
            print("\n--- 正在生成去噪效果对比图 ---")
            example_sample = 'C01_RD15-0027-2_3_4-5;1;1'
            example_marker = 'D8S1179'
            
            example_data = final_long_df_results[
                (final_long_df_results['Sample File'] == example_sample) & 
                (final_long_df_results['Marker'] == example_marker)
            ]
            
            if not example_data.empty:
                raw_data_for_plot = initial_merged_df[
                    (initial_merged_df['Sample File'] == example_sample) & 
                    (initial_merged_df['Marker'] == example_marker)
                ]
                
                fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, sharey=True)
                
                # 子图1: 原始信号
                axes[0].stem(raw_data_for_plot['Raw_Size'], raw_data_for_plot['Raw_Height'],
                             linefmt=f"{CUSTOM_PALETTE['blue']}-", markerfmt=f"o",
                             basefmt=" ", label='原始信号峰')
                axes[0].get_children()[1].get_markerfacecolor().set_alpha(0.7)
                axes[0].get_children()[1].get_markeredgecolor().set_color(CUSTOM_PALETTE['blue'])
                axes[0].set_title(f'去噪效果对比: {example_sample} @ {example_marker} - (1) 原始信号', fontsize=16)
                axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.6)

                # 子图2: 模型去噪后信号
                axes[1].stem(example_data['Size_final'], example_data['Height_final'],
                             linefmt=f"{CUSTOM_PALETTE['red']}-", markerfmt=f"o",
                             basefmt=" ", label='模型去噪后真实峰')
                axes[1].get_children()[1].get_markerfacecolor().set_alpha(0.7)
                axes[1].get_children()[1].get_markeredgecolor().set_color(CUSTOM_PALETTE['red'])
                axes[1].set_title('(2) 模型去噪后信号', fontsize=16)
                axes[1].set_ylabel('峰高 (RFU)', fontsize=14)
                axes[1].legend()
                axes[1].grid(True, linestyle='--', alpha=0.6)

                # 子图3: 真实信号 (附件4)
                axes[2].stem(example_data['Size_final'], example_data['Actual_Denoised_Height_from_Att4'],
                             linefmt=f"{CUSTOM_PALETTE['black']}-", markerfmt=f"o",
                             basefmt=" ", label='真实信号 (Ground Truth)')
                axes[2].get_children()[1].get_markerfacecolor().set_alpha(0.7)
                axes[2].get_children()[1].get_markeredgecolor().set_color(CUSTOM_PALETTE['black'])
                axes[2].set_title('(3) 真实信号 (Ground Truth)', fontsize=16)
                axes[2].set_xlabel('片段大小 (Size)', fontsize=14)
                axes[2].legend()
                axes[2].grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                plt.savefig("denoising_example_comparison.png")
                print("已保存图表: denoising_example_comparison.png")
                plt.show()
            else:
                print(f"警告: 未在结果中找到示例样本 '{example_sample}' @ '{example_marker}'，无法生成对比图。")

            # --- 文件保存 ---
            max_alleles_for_output_format = DEFAULT_MAX_ALLELES_PER_MARKER 
            if processed_data_by_marker: 
                first_marker_key = list(processed_data_by_marker.keys())[0]
                max_alleles_for_output_format = processed_data_by_marker[first_marker_key].get('max_alleles_per_marker', DEFAULT_MAX_ALLELES_PER_MARKER)

            output_wide_df = format_output_like_附件4(final_long_df_results, raw_data_filepath, max_alleles_for_output_format)
            print("\n\n--- 最终输出CSV格式预览 (部分样本) ---")
            print(output_wide_df.head())
            try:
                output_csv_path_att4_like = os.path.join(script_dir, "predicted_peaks_modified_from_附件1.csv")
                output_wide_df.to_csv(output_csv_path_att4_like, index=False, encoding='utf-8-sig')
                print(f"\n修改后的图谱数据已保存到: {output_csv_path_att4_like}")
            except Exception as e: print(f"保存修改后的CSV时发生错误: {e}")
            
            try:
                output_csv_path_long = os.path.join(script_dir, "three_stage_model_predictions_long_final.csv")
                final_long_df_results.to_csv(output_csv_path_long, index=False, encoding='utf-8-sig')
                print(f"\n详细的长格式预测结果已保存到: {output_csv_path_long}")
            except Exception as e: print(f"保存详细长格式预测结果到CSV时发生错误: {e}")
            print("\n--- 任务完成 ---")
        else:
            print("\n模型训练或评估过程中未生成有效结果。")
    else:
        print("\n数据预处理失败，程序终止。")
