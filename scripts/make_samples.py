# scripts/make_samples.py

import numpy as np
import glob
import os
import argparse
import sys

print("--- EXECUTING THIS VERSION OF make_samples.py (FIXED NameError) ---")

# --- process_features function (FIXED) ---
def process_features(files, target_time_length, num_samples_to_take):
    print(f"--- DEBUG [process_features]: Received target_time_length = {target_time_length} ---")
    if not files:
        raise RuntimeError("No .npy files provided to process_features.")

    files_to_process = files[:num_samples_to_take]
    if not files_to_process:
        print(f"Warning: Requested {num_samples_to_take} samples, but only {len(files)} files available. Processing {len(files)} files.")
        if not files: raise RuntimeError(f"No .npy files found to process after slicing to {num_samples_to_take}.")

    # --- 初始化 processed_arrs 在这里 ---
    processed_arrs = []
    actual_processed_count = 0
    # ------------------------------------

    for i, f_path in enumerate(files_to_process):
        try:
            arr = np.load(f_path)
            if arr.ndim != 2 or arr.shape[0] != 40:
                print(f"Warning: Skipping file {f_path}. Expected shape (40, T_i), got {arr.shape}.")
                continue
            original_T = arr.shape[1]
            if original_T == target_time_length:
                processed_arr = arr
            elif original_T < target_time_length:
                padding_width = target_time_length - original_T
                processed_arr = np.pad(arr, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)
            else:
                processed_arr = arr[:, :target_time_length]

            if processed_arr.shape != (40, target_time_length):
                 print(f"--- ERROR [process_features]: Processed shape mismatch for {f_path}! Got {processed_arr.shape}, expected {(40, target_time_length)}")
                 raise RuntimeError("Processed shape mismatch detected inside process_features")

            processed_arrs.append(processed_arr)
            actual_processed_count += 1
        except Exception as e:
            print(f"Warning: Error processing file {f_path}: {e}. Skipping.")
            continue

    # --- 将检查移到循环之后 ---
    if not processed_arrs:
        raise RuntimeError("No features were successfully processed after iterating through files. Check warnings.")
    # ---------------------------

    print(f"Successfully processed {actual_processed_count} features to target length {target_time_length}.")
    batch = np.stack(processed_arrs).astype(np.float32)[:, np.newaxis, :, :]
    return batch

# --- main function (保持之前的重写版本，它看起来没问题) ---
def main():
    parser = argparse.ArgumentParser(description="Create a sample NPY batch from feature files.")
    # ... (argparse 定义不变) ...
    parser.add_argument("--feat_dir", default="data/feats/val/yes", help="Directory containing .wav.npy feature files.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to include in the batch.")
    parser.add_argument("--target_len", type=int, required=True, help="Target time length for each feature.")
    parser.add_argument("--out_dir", default="data", help="Directory to save the output NPY file.")
    parser.add_argument("--out_name", default="sample.npy", help="Name of the output NPY file.")

    try:
        args = parser.parse_args()
        print("--- Parsed Arguments ---")
        print(f"Feature directory: {args.feat_dir}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Target length: {args.target_len}")
        print(f"Output directory: {args.out_dir}")
        print(f"Output filename: {args.out_name}")
        print("------------------------")
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    target_length_from_args = args.target_len
    output_name_from_args = args.out_name
    num_samples_from_args = args.num_samples
    feat_dir_from_args = args.feat_dir
    out_dir_from_args = args.out_dir

    print(f"--- DEBUG: Variables assigned from args ---")
    print(f"target_length_from_args = {target_length_from_args}")
    print(f"output_name_from_args = {output_name_from_args}")
    print("------------------------------------------")

    files = glob.glob(os.path.join(feat_dir_from_args, "*.wav.npy"))
    if not files:
        print(f"Error: No .wav.npy files found in {feat_dir_from_args}")
        sys.exit(1)
    print(f"Found {len(files)} feature files in {feat_dir_from_args}.")

    try:
        print(f"--- DEBUG: Calling process_features with explicit target_len={target_length_from_args} ---")
        batch_data = process_features(files, target_length_from_args, num_samples_from_args)

        print(f"Final batch shape: {batch_data.shape}")
        # --- 更正这里的形状检查逻辑 ---
        actual_processed_count = batch_data.shape[0] # 获取实际处理的样本数
        expected_shape_tuple = (actual_processed_count, 1, 40, target_length_from_args)
        if batch_data.shape != expected_shape_tuple:
             print(f"--- ERROR: Final batch shape {batch_data.shape} does not match expected shape {expected_shape_tuple}! ---")
             # 检查是不是只有样本数不匹配（因为文件不够）
             if batch_data.shape[1:] == (1, 40, target_length_from_args):
                  print(f"--- INFO: Sample count is {actual_processed_count}, possibly less than requested {num_samples_from_args}.")
                  # 如果只是样本数不符，可能还可以继续，取决于你的需求
                  # 如果C,H,W不符，则必须退出
             else:
                  sys.exit(1)
        # --------------------------------

        os.makedirs(out_dir_from_args, exist_ok=True)
        out_path = os.path.join(out_dir_from_args, output_name_from_args)
        print(f"--- DEBUG: Saving to path: {out_path} ---")
        np.save(out_path, batch_data)
        print(f"Saved batch to: {out_path}")

    except Exception as e:
        print(f"An error occurred during processing or saving: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()