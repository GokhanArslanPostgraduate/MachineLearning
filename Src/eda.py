import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV 
import os

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SEED_VALUE = 42

def load_all_datasets():
    print("[1/2] Veri setleri hafızaya yükleniyor...")
    datasets = {}
    gene_mapping = {}
    
    if os.path.exists('data_set_ALL_AML_train.csv') and os.path.exists('actual.csv'):
        train_raw = pd.read_csv('data_set_ALL_AML_train.csv')
        test_raw = pd.read_csv('data_set_ALL_AML_independent.csv')
        actual = pd.read_csv('actual.csv')
        actual_map = actual.set_index('patient')['cancer'].map({'ALL': 0, 'AML': 1})
        actual_map.index = actual_map.index.astype(str).str.strip()
        
        def extract_gene_symbol(desc):
            if pd.isna(desc): return None
            parts = str(desc).split()
            if len(parts) > 0:
                sym = parts[0].strip()
                if not sym.isdigit() and len(sym) > 1:
                    return sym
            return None

        for idx, row in train_raw.iterrows():
            sym = extract_gene_symbol(row.get('Gene Description', ''))
            prob = row['Gene Accession Number']
            gene_mapping[prob] = sym if sym else prob

        def clean_g(df):
            cols = [c for c in df.columns if 'call' not in c.lower() and c != 'Gene Description']
            df_clean = df[cols].set_index('Gene Accession Number').T
            df_clean.index = df_clean.index.astype(str).str.strip()
            df_clean.columns = df_clean.columns.map(gene_mapping)
            return df_clean.loc[:, ~df_clean.columns.duplicated(keep='first')]
            
        X_g = pd.concat([clean_g(train_raw), clean_g(test_raw)], axis=0)
        X_g = X_g.apply(pd.to_numeric, errors='coerce').astype(np.float32)
        X_g.fillna(X_g.median(), inplace=True)
        y_g = X_g.index.map(actual_map).values
        datasets['Golub'] = (X_g, y_g)
        print(f"✅ GOLUB YÜKLENDİ -> Boyut: {X_g.shape}")

    for gse_name in ['GSE6891_GSE26713', 'GSE13159']:
        expr_file = f'{gse_name}_expression_matrix.csv'
        act_file = f'{gse_name}_actual.csv'
        if os.path.exists(expr_file) and os.path.exists(act_file):
            X_raw = pd.read_csv(expr_file, index_col=0).T
            X_raw.index = X_raw.index.astype(str).str.strip()
            X_raw = X_raw.apply(pd.to_numeric, errors='coerce').astype(np.float32)
            X_raw.fillna(X_raw.median(), inplace=True)
            X_raw.columns = X_raw.columns.astype(str).str.split('///').str[0].str.strip()
            X_raw = X_raw.loc[:, ~X_raw.columns.duplicated(keep='first')]
            
            act = pd.read_csv(act_file).set_index('patient')['cancer'].map({'ALL': 0, 'AML': 1})
            act.index = act.index.astype(str).str.strip()
            datasets[gse_name] = (X_raw, X_raw.index.map(act).values)
            print(f"✅ {gse_name} YÜKLENDİ -> Boyut: {X_raw.shape}")
        
    return datasets

def draw_statistical_summary_graphic(stats_summary):
    df_stats = pd.DataFrame(stats_summary)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(df_stats['Dataset']))
    width = 0.35
    
    axes[0].bar(x - width/2, df_stats['ALL_Count'], width, label='ALL', color='royalblue', edgecolor='black')
    axes[0].bar(x + width/2, df_stats['AML_Count'], width, label='AML', color='crimson', edgecolor='black')
    axes[0].set_title("Sınıf Dağılımı (Class Imbalance)", fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_stats['Dataset'], rotation=15)
    axes[0].legend()
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    axes[1].axis('off')
    table_data = [[r['Dataset'], f"{r['Total_Samples']} hasta", f"{r['Gen_Count'][1]} gen", f"%{r['ALL_Perc']:.1f}", f"%{r['AML_Perc']:.1f}"] for _, r in df_stats.iterrows()]
    columns = ["Veri Seti", "Örnek (n)", "Gen (p)", "ALL %", "AML %"]
    table = axes[1].table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_datasets_statistical_summary.png'), dpi=300)
    plt.close()

def pipeline_feature_selection_and_scaling(X_train, y_train, X_val, top_p=200):
    X_train_log = np.log2(np.clip(X_train, 0, None) + 1)
    X_val_log = np.log2(np.clip(X_val, 0, None) + 1)
    
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_log), index=X_train_log.index, columns=X_train_log.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val_log), index=X_val_log.index, columns=X_val_log.columns)
    
    all_idx = np.where(y_train == 0)[0]
    aml_idx = np.where(y_train == 1)[0]
    
    if len(all_idx) >= 2 and len(aml_idx) >= 2:
        _, p_vals = stats.ttest_ind(X_train_imputed.values[all_idx, :], X_train_imputed.values[aml_idx, :], axis=0, equal_var=False)
        p_vals_flat = np.nan_to_num(np.asarray(p_vals).flatten(), nan=1.0)
        top_features_pos = np.argsort(p_vals_flat)[:top_p].flatten()
    else:
        top_features_pos = np.var(X_train_imputed.values, axis=0).argsort()[-top_p:][::-1].flatten()
        
    X_train_opt = X_train_imputed.iloc[:, top_features_pos]
    X_val_opt = X_val_imputed.iloc[:, top_features_pos]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_opt.astype(np.float32))
    X_val_scaled = scaler.transform(X_val_opt.astype(np.float32))
    
    return X_train_scaled, X_val_scaled, top_features_pos

def evaluate_models(datasets):
    print("\n========================================================")
    print("🤖 [2/2] MODELLEME VE ADİL DENEY DÜZENEĞİ BAŞLATILIYOR (SIZINTISIZ)")
    print("========================================================\n")

    report_path = os.path.join(OUTPUT_DIR, 'final_ml_performance_report.csv')
    X_golub, y_golub = datasets['Golub']
    
    models = {
        'Decision Tree (Modül)': DecisionTreeClassifier(max_depth=3, random_state=SEED_VALUE),
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=SEED_VALUE),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED_VALUE),
        'Support Vector Machine': CalibratedClassifierCV(SVC(class_weight='balanced', random_state=SEED_VALUE), ensemble=False)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_VALUE)
    cv_results = {m_name: [] for m_name in models.keys()}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_golub, y_golub)):
        X_train, y_train = X_golub.iloc[train_idx], y_golub[train_idx]
        X_val, y_val = X_golub.iloc[val_idx], y_golub[val_idx]
        X_train_sc, X_val_sc, _ = pipeline_feature_selection_and_scaling(X_train, y_train, X_val)
        
        for m_name, model in models.items():
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_val_sc)
            probs = model.predict_proba(X_val_sc)[:, 1]
            cv_results[m_name].append({
                'Accuracy': accuracy_score(y_val, preds),
                'F1-Score': f1_score(y_val, preds, zero_division=0),
                'AUC': roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
            })
            
    print("📊 --- GOLUB 5-FOLD CROSS-VALIDATION ORTALAMA SONUÇLARI ---")
    for m_name in models.keys():
        df_res = pd.DataFrame(cv_results[m_name])
        means = df_res.mean()
        print(f"📌 {m_name} -> Accuracy: {means['Accuracy']:.3f} | F1-Score: {means['F1-Score']:.3f} | AUC: {means['AUC']:.3f}")
    
    print("[2/2] Bağımsız Açık Kaynak Veri Setleri Üzerinde Genellenebilirlik Test Ediliyor...")
    independent_reports = []
    
    for ind_name in [name for name in datasets.keys() if name != 'Golub']:
        X_ind, y_ind = datasets[ind_name]
        common_genes = list(set(X_golub.columns).intersection(set(X_ind.columns)))
        print(f"   ➔ Golub ve {ind_name} arasındaki ortak gen sayısı: {len(common_genes)}")
        
        if len(common_genes) < 10:
            common_genes = X_golub.columns[:min(200, len(X_ind.columns))].tolist()
            
        X_golub_shared = X_golub[common_genes]
        X_ind_shared = X_ind[common_genes]
        
        X_train_sc, X_test_sc, _ = pipeline_feature_selection_and_scaling(X_golub_shared, y_golub, X_ind_shared)
        
        for m_name, model in models.items():
            model.fit(X_train_sc, y_golub)
            preds = model.predict(X_test_sc)
            independent_reports.append({
                'Test Veri Seti': ind_name, 'Model': m_name,
                'Dış Test Doğruluğu (Accuracy)': accuracy_score(y_ind, preds),
                'Dış Test F1-Skoru': f1_score(y_ind, preds, zero_division=0)
            })
            
    df_ind_results = pd.DataFrame(independent_reports)
    print("📈 --- BAĞIMSIZ VERİ SETİ DIŞ TEST PERFORMANSLARI ---")
    print(df_ind_results.to_string(index=False))
    df_ind_results.to_csv(report_path, index=False)

def main():
    datasets = load_all_datasets()
    if not datasets: return
    
    stats_summary = []
    for name, (X_all, y_all) in datasets.items():
        print(f"🎯 {name} Analiz Ediliyor...")
        X_log = np.log2(np.clip(X_all, 0, None) + 1)
        valid_mask = ~pd.isna(y_all)
        if np.sum(valid_mask) > 0:
            X_log = X_log.iloc[valid_mask, :]
            y_all = y_all[valid_mask].astype(int)
        
        all_c, aml_c = np.sum(y_all == 0), np.sum(y_all == 1)
        total_c = len(y_all)
        stats_summary.append({
            'Dataset': name, 'Total_Samples': total_c, 'Gen_Count': X_all.shape,
            'ALL_Count': all_c, 'AML_Count': aml_c,
            'ALL_Perc': (all_c / total_c) * 100, 'AML_Perc': (aml_c / total_c) * 100
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sample_genes = X_all.columns[:min(100, len(X_all.columns))]
        raw_vals = X_all.iloc[valid_mask][sample_genes].values.flatten()
        log_vals = X_log[sample_genes].values.flatten()
        
        axes[0].hist(raw_vals[~np.isnan(raw_vals)], bins=50, density=True, color='crimson', alpha=0.7, edgecolor='black')
        axes[0].set_title(f"{name} Ham Sinyal")
        axes[1].hist(log_vals[~np.isnan(log_vals)], bins=50, density=True, color='royalblue', alpha=0.7, edgecolor='black')
        axes[1].set_title("Log2 Dağılımı")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_eda_1_log_transformation.png'), dpi=300)
        plt.close()

        X_train_sc, X_val_sc, top_200_pos = pipeline_feature_selection_and_scaling(X_log, y_all, X_log)
        
        # PCA Modelinin kurulması ve varyans oranlarının hesaplanması
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_sc)
        exp_var = pca.explained_variance_ratio_ * 100  # Yüzdelik formata çevirme
        
        # Terminale varyans oranlarını basma
        print(f"   ➔ {name} PCA Varyans Oranları -> PC1: %{exp_var[0]:.2f} | PC2: %{exp_var[1]:.2f} | Toplam: %{sum(exp_var):.2f}")
        
        # t-SNE Modelinin kurulması
        tsne = TSNE(n_components=2, perplexity=min(15, max(5, len(X_log)//4)), random_state=SEED_VALUE)
        X_tsne = tsne.fit_transform(X_train_sc)
        
        # Grafiklerin Çizilmesi
        fig, axes_dim = plt.subplots(1, 2, figsize=(15, 6))
        colors, target_names, markers = {0: 'royalblue', 1: 'crimson'}, {0: 'ALL', 1: 'AML'}, {0: 'o', 1: 'X'}
        
        for lbl in np.unique(y_all):
            if lbl in target_names:
                mask = (y_all == lbl)
                axes_dim[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[lbl], label=target_names[lbl], marker=markers[lbl], s=100, edgecolors='black', alpha=0.8)
                axes_dim[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[lbl], label=target_names[lbl], marker=markers[lbl], s=100, edgecolors='black', alpha=0.8)
        
        # PCA Grafiği Başlık ve Dinamik Eksen Etiketleri
        axes_dim[0].set_title(f"{name} PCA Analizi", fontweight='bold')
        axes_dim[0].set_xlabel(f"Temel Bileşen 1 (Açıklanan Varyans: %{exp_var[0]:.1f})")
        axes_dim[0].set_ylabel(f"Temel Bileşen 2 (Açıklanan Varyans: %{exp_var[1]:.1f})")
        axes_dim[0].legend()
        axes_dim[0].grid(True, linestyle='--', alpha=0.5)
        
        # t-SNE Grafiği Başlık ve Eksen Etiketleri
        axes_dim[1].set_title(f"{name} t-SNE Analizi", fontweight='bold')
        axes_dim[1].set_xlabel("t-SNE Boyut 1")
        axes_dim[1].set_ylabel("t-SNE Boyut 2")
        axes_dim[1].legend()
        axes_dim[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_eda_2_dimension_reduction_optimized.png'), dpi=300)
        plt.close()

        try:
            fig = plt.figure(figsize=(12, 9))
            top_30_pos = top_200_pos[:30]
            heatmap_data = X_log.iloc[:, top_30_pos].T
            heatmap_scaled = StandardScaler().fit_transform(heatmap_data.T).T
            linkage_matrix = linkage(heatmap_scaled, method='ward')
            ax_dendro = fig.add_axes([0.05, 0.1, 0.2, 0.7])
            dendrogram(linkage_matrix, orientation='left', ax=ax_dendro, no_labels=True)
            ax_dendro.axis('off')
            ax_matrix = fig.add_axes([0.3, 0.1, 0.55, 0.7])
            im = ax_matrix.imshow(heatmap_scaled, aspect='auto', cmap='bwr', vmin=-2, vmax=2)
            ax_matrix.set_yticks(range(30))
            ax_matrix.set_yticklabels(heatmap_data.index, fontsize=8)
            plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_eda_3_hierarchical_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except:
            pass

        dt_model = DecisionTreeClassifier(max_depth=3, random_state=SEED_VALUE)
        dt_model.fit(X_train_sc, y_all)
        plt.figure(figsize=(16, 8))
        plot_tree(dt_model, feature_names=X_log.iloc[:, top_200_pos].columns.tolist(), class_names=['ALL', 'AML'], filled=True, rounded=True, fontsize=10)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_decision_tree_output.png'), dpi=300)
        plt.close()
        

    draw_statistical_summary_graphic(stats_summary)
    
    report_path = os.path.join(OUTPUT_DIR, 'final_ml_performance_report.csv')
    if os.path.exists(report_path):
        os.remove(report_path)
        
    evaluate_models(datasets)

if __name__ == '__main__':
    main()