
"""Predict genotype-by-environment phenotypes using PCA and BiLSTM features.

Evaluate regression models under the supplied cross-validation schemes and
export predictions, feature weights, and Pearson correlation summaries.
"""

# Data loading and environment configuration
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time
import warnings
import argparse
import pickle
import gc  # Garbage collection for large intermediate arrays.
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# GBLUP regression with REML variance-ratio estimation
class GBLUPRegressor(BaseEstimator, RegressorMixin):
    """Fit GBLUP using a profiled REML estimate of the variance ratio.

    Model: y = 1 * mu + u + e, with covariance sigma_g^2 * K for u
    and sigma_e^2 * I for e. Construct K = Z @ Z.T / m from standardized
    features. After eigendecomposition, each likelihood evaluation is O(n).
    Recover coefficients on the original input scale via RR-BLUP equivalence.
    """
    def __init__(self, bounds=(1e-5, 1e5)):
        self.bounds = bounds

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, m = X.shape

        # Standardize features and construct K = Z @ Z.T / m.
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # Avoid division by zero for constant features.
        Z = (X - self.mean_) / self.std_
        self.m_ = m

        K = np.dot(Z, Z.T) / m

        # Eigendecompose the symmetric relationship matrix.
        d, U = np.linalg.eigh(K)
        d = np.maximum(d, 0.0)  # Clip small negative eigenvalues caused by numerical error.

        Uy = np.dot(U.T, y)
        U1 = np.dot(U.T, np.ones(n))

        # Profile REML over log(lambda), where lambda = sigma_e^2 / sigma_g^2.
        def neg_reml(log_lam):
            """Evaluate the profiled negative REML log-likelihood at log(lambda)."""
            lam = np.exp(log_lam)
            v = d + lam  # Eigenvalues of K + lambda * I.

            # Estimate the intercept by generalized least squares.
            denom = np.sum((U1 ** 2) / v)
            mu = np.sum(U1 * Uy / v) / denom

            res = Uy - U1 * mu
            yPy = np.sum((res ** 2) / v)
            if yPy <= 0:
                return 1e10

            log_det_V = np.sum(np.log(v))
            log_det_XVX = np.log(denom)

            # Profile negative REML log-likelihood, up to additive constants.
            return 0.5 * (log_det_V + log_det_XVX + (n - 1) * np.log(yPy))

        # Optimize the variance ratio on the logarithmic scale.
        opt_res = minimize_scalar(
            neg_reml,
            bounds=(np.log(self.bounds[0]), np.log(self.bounds[1])),
            method='bounded'
        )
        self.lambda_ = np.exp(opt_res.x)

        # Compute the fitted intercept and alpha = (K + lambda * I)^-1 (y - mu).
        v_opt = d + self.lambda_
        self.mu_ = np.sum(U1 * Uy / v_opt) / np.sum((U1 ** 2) / v_opt)
        res_opt = Uy - U1 * self.mu_
        alpha = np.dot(U, res_opt / v_opt)

        # Recover feature effects using the equivalence of GBLUP and RR-BLUP.
        # u_test = K_{test, train} * alpha = (Z_test * Z_train^T / m) * alpha = Z_test * (Z_train^T * alpha / m)
        beta_std = np.dot(Z.T, alpha) / m

        # Restore the original feature scale for prediction and PCA back-projection.
        self.coef_ = beta_std / self.std_
        self.intercept_ = self.mu_ - np.dot(self.mean_, self.coef_)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.coef_) + self.intercept_


# Command-line arguments
parser = argparse.ArgumentParser(description="GxE Prediction ML Pipeline (Direct Concatenation)")
parser.add_argument('--geno_path', type=str, required=True, help="Path to genotype CSV file")
parser.add_argument('--env_path', type=str, required=True, help="Path to environment CSV file")
parser.add_argument('--pheno_path', type=str, required=True, help="Path to phenotype CSV file")
parser.add_argument('--pheno_name', type=str, required=True, help="Trait prefix, e.g., CL or HD")
parser.add_argument('--cvf_path', type=str, required=True, help="Directory containing CV files")
parser.add_argument('--model_path', type=str, default='./', help="Output directory")

# Run one model or all available models.
parser.add_argument('--model_type', type=str, default='All',
                    choices=['All', 'LightGBM', 'XGBoost', 'RandomForest', 'Ridge', 'Lasso', 'GBLUP'],
                    help="Select a specific model to run, or 'All'.")

# Genotype and environment feature dimensions.
parser.add_argument('--pca_dim', type=float, default=0.95, help="PCA components (>1) OR variance ratio (<=1.0)")
parser.add_argument('--env_dim', type=int, default=128, help="Fixed dimension for Environment features via BiLSTM")
parser.add_argument('--cache_dir', type=str, default='./cache')
args = parser.parse_args()

# Input paths and output directories
GENO_PATH = args.geno_path
ENV_PATH = args.env_path
PHENO_PATH = args.pheno_path
TARGET_TRAIT = args.pheno_name
OUT_DIR = args.model_path
CACHE_DIR = args.cache_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

WEIGHTS_DIR = os.path.join(OUT_DIR, 'Feature_Weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Resolve CV files with CSV, TXT, or unsuffixed names.
def get_cv_path(base_dir, base_name):
    """Resolve a CV path, preferring a CSV file, then TXT, then the base name."""
    if os.path.exists(os.path.join(base_dir, f"{base_name}.csv")):
        return os.path.join(base_dir, f"{base_name}.csv")
    elif os.path.exists(os.path.join(base_dir, f"{base_name}.txt")):
        return os.path.join(base_dir, f"{base_name}.txt")
    return os.path.join(base_dir, base_name)

CV_FILES = {
    'CV1_Unknown_G': get_cv_path(args.cvf_path, 'CV1_Unknown_Genotypes'),
    'CV2_Unknown_E': get_cv_path(args.cvf_path, 'CV2_Unknown_Environments'),
    'CV3_Unknown_GxE': get_cv_path(args.cvf_path, 'CV3_Strict_Splits')
}

# Fixed-dimensional environment feature extraction
class EnvBiLSTMAutoencoder(nn.Module):
    """Encode environment time series with a bidirectional LSTM autoencoder."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        h_rep = h_cat.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h_rep)
        return out, h_cat

def extract_env_features(env_path, cache_dir, target_dim):
    """Load cached environment embeddings or train and cache a BiLSTM autoencoder."""
    cache_file = os.path.join(cache_dir, f'env_features_fixed_{target_dim}D.pkl')

    if os.path.exists(cache_file):
        print(f"📂 发现固定的 [{target_dim}维] 环境特征缓存，直接加载...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n🌤️ [步骤 1] 正在使用 BiLSTM 提取环境特征，目标固定维度: {target_dim} 维...")

    try:
        df_env = pd.read_csv(env_path, sep='\t')
        if 'Env' not in df_env.columns:
            df_env = pd.read_csv(env_path)
    except Exception:
        df_env = pd.read_csv(env_path)

    envs = df_env['Env'].unique()
    feature_cols = df_env.columns[2:]

    env_tensors = []
    env_names = []

    for env in envs:
        env_data = df_env[df_env['Env'] == env][feature_cols].values
        env_data = (env_data - env_data.mean(axis=0)) / (env_data.std(axis=0) + 1e-8)
        env_tensors.append(torch.tensor(env_data, dtype=torch.float32))
        env_names.append(env)

    max_len = max([t.size(0) for t in env_tensors])
    padded_tensors = [torch.cat([t, torch.zeros(max_len - t.size(0), t.size(1))]) for t in env_tensors]
    batch_x = torch.stack(padded_tensors)

    input_dim = batch_x.size(-1)
    hidden_dim = (target_dim + 1) // 2

    model = EnvBiLSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        recon, _ = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, h_cat = model(batch_x)
        h_aligned = h_cat[:, :target_dim]

    print(f"   - 🔍 BiLSTM 提取完成，最终环境维度: {h_aligned.shape[1]} 维")

    env_dict = {name: h_aligned[i].numpy() for i, name in enumerate(env_names)}
    with open(cache_file, 'wb') as f:
        pickle.dump(env_dict, f)

    del model, batch_x, env_tensors
    gc.collect() # Release intermediate memory.
    return env_dict


# Feature preprocessing and direct concatenation
def main():
    # Skip the task when its summary file already exists.
    summary_filename = f"Summary_{args.model_type}_PCA{args.pca_dim}_Env{args.env_dim}.csv"
    summary_filepath = os.path.join(OUT_DIR, summary_filename)
    if os.path.exists(summary_filepath):
        print(f"✅ 检测到目标汇总文件 {summary_filepath} 已存在，直接跳过任务！")
        return  # Return without recomputing completed results.

    # Compute features only for unfinished tasks.
    env_dict = extract_env_features(ENV_PATH, CACHE_DIR, target_dim=args.env_dim)

    n_components = args.pca_dim if args.pca_dim <= 1.0 else int(args.pca_dim)
    strategy_str = f"保留 {int(n_components*100)}% 方差" if n_components <= 1.0 else f"固定 {n_components} 维"

    geno_cache_base = os.path.join(CACHE_DIR, f'geno_pca_{args.pca_dim}')
    geno_ids_file = geno_cache_base + '_ids.npy'
    geno_feat_file = geno_cache_base + '_feat.npy'
    geno_comp_file = geno_cache_base + '_comp.npy'
    geno_snps_file = geno_cache_base + '_snps.pkl'

    # Load cached PCA features or fit and save the transformation.
    if os.path.exists(geno_ids_file) and os.path.exists(geno_feat_file) and os.path.exists(geno_comp_file) and os.path.exists(geno_snps_file):
        print(f"\n🧬 [步骤 2] 发现策略 [{strategy_str}] 的基因型 PCA 缓存，直接轻量级加载...")
        g_ids = np.load(geno_ids_file, allow_pickle=True)
        g_features_pca = np.load(geno_feat_file, allow_pickle=True)

        with open(geno_snps_file, 'rb') as f:
            snp_names = pickle.load(f)

        g_dict = {g_ids[i]: g_features_pca[i] for i in range(len(g_ids))}
        actual_g_dim = g_features_pca.shape[1]

    else:
        print(f"\n🧬 [步骤 2] 正在加载基因型数据并进行 PCA 降维...")
        print(f"   👉 设定的 PCA 策略为: {strategy_str}")

        df_g = pd.read_csv(GENO_PATH)
        g_ids = df_g.iloc[:, 0].values
        snp_names = df_g.columns[1:].tolist()
        g_features = df_g.iloc[:, 1:].values

        np.save(geno_ids_file, g_ids)
        with open(geno_snps_file, 'wb') as f:
            pickle.dump(snp_names, f)

        del df_g
        gc.collect()

        pca = PCA(n_components=n_components, random_state=42)
        g_features_pca = pca.fit_transform(g_features)

        np.save(geno_feat_file, g_features_pca)
        np.save(geno_comp_file, pca.components_)

        g_dict = {g_ids[i]: g_features_pca[i] for i in range(len(g_ids))}
        actual_g_dim = g_features_pca.shape[1]

        del g_features, pca
        gc.collect()

    print(f"✅ PCA 处理完毕！基因型(G) 实际被压缩至: {actual_g_dim} 维。巨大的载荷矩阵已落盘。")

    # ----------------------------------------------------
    print("\n🌾 [步骤 3] 正在构建隐式交互融合矩阵 [G, E] ...")

    try:
        df_p = pd.read_csv(PHENO_PATH, sep='\t', encoding='utf-8')
        if 'ID' not in df_p.columns:
            df_p = pd.read_csv(PHENO_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df_p = pd.read_csv(PHENO_PATH, encoding='gbk')

    pheno_cols = [c for c in df_p.columns if c.startswith(TARGET_TRAIT + '_')]

    dataset_records = []
    for idx, row in df_p.iterrows():
        base_id = str(row['ID'])  # Convert phenotype identifiers to strings before lookup.
        if base_id not in g_dict:
            continue
        g_vec = g_dict[base_id]

        for p_col in pheno_cols:
            # Use the final underscore-delimited field as the environment identifier.
            env_name = p_col.split('_')[-1]

            raw_val = row[p_col]

            # Skip values that cannot be converted to numeric phenotypes.
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                continue

            if pd.isna(val) or env_name not in env_dict:
                continue

            e_vec = env_dict[env_name]
            fused_features = np.concatenate([g_vec, e_vec])

            dataset_records.append({
                'ID_Env': f"{base_id}_{env_name}",
                'Trait': val,
                'Features': fused_features
            })

    if len(dataset_records) == 0:
        print(f"\n🚨 严重错误：匹配后数据量为 0！")
        print(f"👉 请检查: 表型列 {pheno_cols[:3]} 提取出的年份是否在环境文件 {list(env_dict.keys())[:3]} 中！")
        sys.exit(1)

    df_dataset = pd.DataFrame(dataset_records)
    X_all = np.vstack(df_dataset['Features'].values)
    y_all = df_dataset['Trait'].values
    id_all = df_dataset['ID_Env'].values

    # Replace NaN and infinite feature values with zero before regression.
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    del df_dataset, dataset_records
    gc.collect()

    print(f"✅ 融合矩阵构建完成！有效样本总数: {len(X_all)}")
    print(f"   - 🚀 输入给 ML 模型的最终特征总维度: {X_all.shape[1]} 维")

    feature_names_compressed = [f"G_PC_{i+1}" for i in range(actual_g_dim)] + [f"E_Embed_{i+1}" for i in range(args.env_dim)]

    # Model training and cross-validation
    all_models = {
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=500, tol=1e-3),
        'GBLUP': GBLUPRegressor()
    }

    if args.model_type == 'All':
        models_to_run = all_models
    else:
        models_to_run = {args.model_type: all_models[args.model_type]}

    summary_results = []

    for cv_name, cv_path in CV_FILES.items():
        if not os.path.exists(cv_path):
            print(f"\n⚠️ 警告: 找不到路径 {cv_path}，跳过 {cv_name}。")
            continue

        print(f"\n{'='*20} {cv_name} {'='*20}")

        for model_name, model in models_to_run.items():
            pred_out_path = os.path.join(OUT_DIR, f"{cv_name}_{model_name}_predictions.csv")
            file1_path = os.path.join(WEIGHTS_DIR, f"{cv_name}_{model_name}_compressed_features_weights.csv")
            file2_path = os.path.join(WEIGHTS_DIR, f"{cv_name}_{model_name}_original_SNP_weights.csv")

            # Reuse results when predictions and both feature-weight files exist.
            if os.path.exists(pred_out_path) and os.path.exists(file1_path) and os.path.exists(file2_path):
                print(f"  > 检测到 {model_name} 结果文件已存在，直接跳过训练！", end=" ")
                try:
                    df_exist = pd.read_csv(pred_out_path)
                    if len(df_exist) > 1 and np.std(df_exist['Predicted']) > 0 and np.std(df_exist['Actual']) > 0:
                        mean_r = pearsonr(df_exist['Predicted'], df_exist['Actual'])[0]
                    else:
                        mean_r = 0.0
                    print(f"(已存结果 PCC: {mean_r:.4f})")
                    summary_results.append({
                        'CV_Scheme': cv_name,
                        'Model': model_name,
                        'Mean_Pearson': mean_r
                    })
                    continue
                except Exception as e:
                    print(f"\n    ⚠️ 读取已有结果文件失败 ({e})，将重新运行 {model_name} ...")

            print(f"  > 运行 {model_name} ...", end=" ", flush=True)
            all_preds, all_trues, all_names, fold_pearsons = [], [], [], []
            all_fold_weights = []
            t0 = time.time()

            if os.path.isdir(cv_path):
                test_files = [f for f in os.listdir(cv_path) if f.endswith('_Test.txt') or f.endswith('_Test.csv')]
                k_folds = len(test_files)
                for f in range(1, k_folds + 1):
                    train_file = os.path.join(cv_path, f'Fold_{f}_Train.txt')
                    test_file = os.path.join(cv_path, f'Fold_{f}_Test.txt')
                    if not os.path.exists(train_file):
                        train_file = train_file.replace('.txt', '.csv')
                        test_file = test_file.replace('.txt', '.csv')
                    if not os.path.exists(train_file) or not os.path.exists(test_file): continue

                    df_train = pd.read_csv(train_file)
                    df_test = pd.read_csv(test_file)
                    train_ids_set = set(df_train.iloc[:, 0].astype(str))
                    test_ids_set = set(df_test.iloc[:, 0].astype(str))

                    train_mask = np.array([uid in train_ids_set for uid in id_all])
                    test_mask = np.array([uid in test_ids_set for uid in id_all])

                    X_train, y_train = X_all[train_mask], y_all[train_mask]
                    X_test, y_test = X_all[test_mask], y_all[test_mask]
                    test_ids = id_all[test_mask]

                    if len(X_train) == 0 or len(X_test) == 0: continue

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    all_preds.extend(preds)
                    all_trues.extend(y_test)
                    all_names.extend(test_ids)

                    if np.std(preds) > 0 and np.std(y_test) > 0:
                        fold_pearsons.append(pearsonr(preds, y_test)[0])
                    else:
                        fold_pearsons.append(0.0)

                    if hasattr(model, 'feature_importances_'):
                        all_fold_weights.append(model.feature_importances_)
                    elif hasattr(model, 'coef_'):
                        all_fold_weights.append(model.coef_)
            else:
                try:
                    df_cv = pd.read_csv(cv_path)
                    if 'ID' not in df_cv.columns:
                        df_cv = pd.read_csv(cv_path, sep='\t')
                except Exception:
                    df_cv = pd.read_csv(cv_path, sep='\t')

                cv_dict = dict(zip(df_cv.iloc[:, 0].astype(str), df_cv.iloc[:, 1]))
                folds = np.array([cv_dict.get(uid, -1) for uid in id_all])
                valid_mask = folds != -1
                X_cv, y_cv, id_cv, folds_cv = X_all[valid_mask], y_all[valid_mask], id_all[valid_mask], folds[valid_mask]
                unique_folds = np.unique(folds_cv)

                for f in unique_folds:
                    if f == 0: continue
                    train_idx = folds_cv != f
                    test_idx = folds_cv == f
                    X_train, X_test = X_cv[train_idx], X_cv[test_idx]
                    y_train, y_test = y_cv[train_idx], y_cv[test_idx]
                    if len(X_train) == 0 or len(X_test) == 0: continue

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    all_preds.extend(preds)
                    all_trues.extend(y_test)
                    all_names.extend(id_cv[test_idx])

                    if np.std(preds) > 0 and np.std(y_test) > 0:
                        fold_pearsons.append(pearsonr(preds, y_test)[0])
                    else:
                        fold_pearsons.append(0.0)

                    if hasattr(model, 'feature_importances_'):
                        all_fold_weights.append(model.feature_importances_)
                    elif hasattr(model, 'coef_'):
                        all_fold_weights.append(model.coef_)

            # ----------------------------------------------------
            t_cost = time.time() - t0
            mean_r = np.mean(fold_pearsons) if len(fold_pearsons) > 0 else 0
            print(f"PCC: {mean_r:.4f} | 耗时: {t_cost:.1f}s")

            summary_results.append({
                'CV_Scheme': cv_name,
                'Model': model_name,
                'Mean_Pearson': mean_r
            })

            df_pred_out = pd.DataFrame({'Sample_ID': all_names, 'Actual': all_trues, 'Predicted': all_preds})
            df_pred_out.to_csv(pred_out_path, index=False)

            # Feature-weight export
            if all_fold_weights:
                avg_weights = np.mean(all_fold_weights, axis=0)

                # Save weights or importances for compressed genotype and environment features.
                df_compressed = pd.DataFrame({
                    'Feature': feature_names_compressed,
                    'Weight_or_Importance': avg_weights
                })

                if hasattr(model, 'coef_'):
                    df_compressed['Abs_Weight'] = df_compressed['Weight_or_Importance'].abs()
                    df_compressed = df_compressed.sort_values(by='Abs_Weight', ascending=False)
                else:
                    df_compressed = df_compressed.sort_values(by='Weight_or_Importance', ascending=False)

                df_compressed.to_csv(file1_path, index=False)

                # Memory-map PCA loadings to project genotype weights to SNP features.
                g_pc_weights = avg_weights[:actual_g_dim]

                pca_components_mmap = np.load(geno_comp_file, mmap_mode='r')

                if hasattr(model, 'coef_'):
                    original_snp_weights = np.dot(g_pc_weights, pca_components_mmap)
                else:
                    original_snp_weights = np.dot(g_pc_weights, np.abs(pca_components_mmap))

                del pca_components_mmap
                gc.collect()

                df_snp = pd.DataFrame({
                    'SNP_Name': snp_names,
                    'Weight_or_Importance': original_snp_weights
                })

                if hasattr(model, 'coef_'):
                    df_snp['Abs_Weight'] = df_snp['Weight_or_Importance'].abs()
                    df_snp = df_snp.sort_values(by='Abs_Weight', ascending=False)
                else:
                    df_snp = df_snp.sort_values(by='Weight_or_Importance', ascending=False)

                df_snp.to_csv(file2_path, index=False)

                del df_compressed, df_snp, original_snp_weights, avg_weights
                gc.collect()

    # Cross-validation summary
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(summary_filepath, index=False)

        print(f"\n🎉 所有模型评估完毕！特征权重已保存至目录: {WEIGHTS_DIR}")
        print("-" * 50)
        print(df_summary.pivot(index='Model', columns='CV_Scheme', values='Mean_Pearson'))
        print("-" * 50)
    else:
        print("\n⚠️ 没有成功运行任何 CV 划分，请检查路径。")

if __name__ == '__main__':
    main()
