import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED_VALUE = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

DATASET_NAMES = [
    "Golub",
    "GSE6891_GSE26713",
    "GSE13159"
]

def load_golub():

    train_raw = pd.read_csv(
        "data_set_ALL_AML_train.csv"
    )

    test_raw = pd.read_csv(
        "data_set_ALL_AML_independent.csv"
    )

    actual = pd.read_csv("actual.csv")

    actual_map = (
        actual.set_index("patient")["cancer"]
        .map({"ALL": 0, "AML": 1})
    )

    actual_map.index = (
        actual_map.index.astype(str).str.strip()
    )

    def extract_samples(df):

        cols = [
            c for c in df.columns
            if "call" not in c.lower()
            and c != "Gene Description"
        ]

        X = df[cols].copy()

        X = X.set_index(
            "Gene Accession Number"
        ).T

        X.index = (
            X.index.astype(str).str.strip()
        )

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        return X

    X_train = extract_samples(train_raw)
    X_test = extract_samples(test_raw)

    # Prob kimliklerini koruyarak birleştir.
    X = pd.concat(
        [X_train, X_test],
        axis=0
    )

    y = X.index.map(actual_map)

    # Anotasyon eşleştirmesi
    annotation = pd.read_csv(
        "golub_probe_annotation.csv"
    )

    probe_col = "Gene Accession Number"
    symbol_col = "Gene Symbol"

    annotation = annotation[
        [probe_col, symbol_col]
    ].dropna()

    annotation[probe_col] = (
        annotation[probe_col].astype(str)
    )

    annotation[symbol_col] = (
        annotation[symbol_col]
        .astype(str)
        .str.strip()
    )

    mapping = dict(
        zip(
            annotation[probe_col],
            annotation[symbol_col]
        )
    )

    X.columns = [
        mapping.get(str(c), str(c))
        for c in X.columns
    ]

    X = X.loc[
        :, X.columns.notna()
    ]

    X = X.loc[
        :, ~X.columns.duplicated()
    ]

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    return X, np.asarray(y), {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "normalization": "Raw microarray signals"
    }

def load_gse(name, normalization):

    expression_file = (
        f"{name}_expression_matrix.csv"
    )

    actual_file = (
        f"{name}_actual.csv"
    )

    X = pd.read_csv(
        expression_file,
        index_col=0
    ).T

    X.index = (
        X.index.astype(str).str.strip()
    )

    X.columns = (
        X.columns.astype(str)
        .str.split("///")
        .str[0]
        .str.strip()
    )

    X = X.loc[
        :, ~X.columns.duplicated()
    ]

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    actual = pd.read_csv(
        actual_file
    )

    actual_map = (
        actual.set_index("patient")["cancer"]
        .map({"ALL": 0, "AML": 1})
    )

    actual_map.index = (
        actual_map.index.astype(str).str.strip()
    )

    y = X.index.map(actual_map)

    return X, np.asarray(y), {
        "normalization": normalization
    }

def load_all():

    datasets = {}

    datasets["Golub"] = load_golub()

    datasets["GSE6891_GSE26713"] = load_gse(
        "GSE6891_GSE26713",
        "MAS5 / RMA (ayrı kaynak bileşenleri)"
    )

    datasets["GSE13159"] = load_gse(
        "GSE13159",
        "MAS5"
    )

    return datasets

def data_quality(X, y, name):

    print("\n", "=" * 60)
    print(name, "VERİ KALİTESİ")
    print("=" * 60)

    valid = pd.notna(y)

    X_valid = X.loc[valid]
    y_valid = np.asarray(y)[valid]

    report = {
        "Dataset": name,
        "Samples": len(X),
        "Genes": X.shape[1],
        "Missing": int(X.isna().sum().sum()),
        "INF": int(
            np.isinf(
                X.select_dtypes(
                    include=np.number
                ).values
            ).sum()
        ),
        "Negative": int(
            (X.select_dtypes(
                include=np.number
            ) < 0).sum().sum()
        ),
        "Unmatched_labels": int(
            pd.isna(y).sum()
        ),
        "ALL": int(np.sum(y_valid == 0)),
        "AML": int(np.sum(y_valid == 1))
    }

    print(pd.Series(report))

    return report

def descriptive_statistics(X, name):

    desc = X.describe().T

    desc.to_csv(
        f"{OUTPUT_DIR}/{name}_gene_statistics.csv"
    )

    sample_stats = X.describe().T

    print(f"\n{name} betimsel istatistikleri:")
    print(sample_stats[
        ["mean", "std", "min", "max"]
    ].describe())

    return desc

def expression_distribution(X, name):

    values = X.values.flatten()

    values = values[
        np.isfinite(values)
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=100,
        color="royalblue",
        alpha=0.75
    )

    plt.title(
        f"{name} - Mevcut İfade Dağılımı"
    )

    plt.xlabel("Expression value")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{name}_distribution.png",
        dpi=300
    )

    plt.close()

def class_distribution(y, name):

    valid_y = np.asarray(y)
    valid_y = valid_y[pd.notna(valid_y)].astype(int)

    counts = pd.Series(
        valid_y
    ).map({
        0: "ALL",
        1: "AML"
    }).value_counts()

    plt.figure(figsize=(7, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=["royalblue", "crimson"],
        legend=False
    )

    plt.title(f"{name} - Sınıf Dağılımı")
    plt.ylabel("Sample count")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{name}_classes.png",
        dpi=300
    )

    plt.close()

def dimensionality_reduction(X, y, name):

    valid = pd.notna(y)

    X = X.loc[valid].copy()

    y = np.asarray(y)[valid].astype(int)

    # Görselleştirme için eksik değer doldurma.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    # Sabit özellikleri kaldır.
    X = X.loc[
        :, X.nunique() > 1
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(
        X_scaled
    )

    variance = (
        pca.explained_variance_ratio_ * 100
    )

    print(
        f"{name} PCA: "
        f"PC1 %{variance[0]:.2f} | "
        f"PC2 %{variance[1]:.2f} | "
        f"Toplam %{variance.sum():.2f}"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    colors = {
        0: "royalblue",
        1: "crimson"
    }

    labels = {
        0: "ALL",
        1: "AML"
    }

    for cls in [0, 1]:

        mask = y == cls

        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[cls],
            label=labels[cls],
            alpha=0.8
        )

    axes[0].set_title(
        f"PCA - {name}"
    )

    axes[0].set_xlabel(
        f"PC1 (%{variance[0]:.2f})"
    )

    axes[0].set_ylabel(
        f"PC2 (%{variance[1]:.2f})"
    )

    axes[0].legend()

    # t-SNE
    if len(X_scaled) >= 6:

        perplexity = min(
            15,
            len(X_scaled) - 1
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=SEED_VALUE,
            init="pca",
            learning_rate="auto"
        )

        X_tsne = tsne.fit_transform(
            X_scaled
        )

        for cls in [0, 1]:

            mask = y == cls

            axes[1].scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=colors[cls],
                label=labels[cls],
                alpha=0.8
            )

        axes[1].set_title(
            f"t-SNE - {name}"
        )

        axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{name}_PCA_TSNE.png",
        dpi=300
    )

    plt.close()

def differential_gene_analysis(X, y, name):

    valid = pd.notna(y)

    X = X.loc[valid]

    y = np.asarray(y)[valid].astype(int)

    all_idx = y == 0
    aml_idx = y == 1

    results = []

    for gene in X.columns:

        all_values = X.loc[
            all_idx, gene
        ].dropna()

        aml_values = X.loc[
            aml_idx, gene
        ].dropna()

        if (
            len(all_values) < 2
            or len(aml_values) < 2
        ):
            continue

        _, p_value = stats.ttest_ind(
            all_values,
            aml_values,
            equal_var=False
        )

        results.append({
            "Gene": gene,
            "ALL_Mean": all_values.mean(),
            "AML_Mean": aml_values.mean(),
            "Mean_Difference": (
                aml_values.mean()
                - all_values.mean()
            ),
            "P_Value": p_value
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # Benjamini-Hochberg FDR düzeltmesi.
    from statsmodels.stats.multitest import multipletests

    result_df["Adjusted_P"] = multipletests(
        result_df["P_Value"],
        method="fdr_bh"
    )[1]

    result_df = result_df.sort_values(
        "P_Value"
    )

    result_df.to_csv(
        f"{OUTPUT_DIR}/{name}_differential_genes.csv",
        index=False
    )

    return result_df

def create_heatmap(X, y, gene_results, name):

    if gene_results.empty:
        return

    top_genes = gene_results.head(30)["Gene"].tolist()

    valid = pd.notna(y)

    X_plot = X.loc[
        valid, top_genes
    ].copy()

    y_plot = np.asarray(y)[valid]

    X_plot = X_plot.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_plot = X_plot.fillna(
        X_plot.median()
    )

    # Gen bazında örnekler üzerinden standardizasyon.
    X_scaled = StandardScaler().fit_transform(
        X_plot
    )

    X_scaled = pd.DataFrame(
        X_scaled,
        index=X_plot.index,
        columns=top_genes
    )

    plt.figure(figsize=(14, 9))

    sns.clustermap(
        X_scaled.T,
        cmap="bwr",
        center=0,
        vmin=-2,
        vmax=2,
        col_cluster=True,
        row_cluster=True,
        figsize=(14, 10)
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{name}_heatmap.png",
        dpi=300
    )

    plt.close("all")

def evaluate_model(model, X_test, y_test, dataset_name, positive_label=1):

    # Tahminler
    y_pred = model.predict(X_test)

    # Sınıf etiketleri
    classes = np.unique(y_test)

    # Temel metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred,
        pos_label=positive_label,
        zero_division=0
    )
    recall = recall_score(
        y_test, y_pred,
        pos_label=positive_label,
        zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred,
        pos_label=positive_label,
        zero_division=0
    )

    # ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            positive_index = list(model.classes_).index(positive_label)
            auc = roc_auc_score(
                y_test,
                probabilities[:, positive_index]
            )
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            auc = roc_auc_score(
                y_test,
                scores,
                pos_label=positive_label
            )
        else:
            auc = np.nan
    except (ValueError, IndexError):
        auc = np.nan

    # Sonuç tablosu
    results = {
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": auc
    }

    print(f"\n--- {dataset_name} ---")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return results

def train_and_evaluate_models(X, y, metadata):

    print("\n" + "=" * 60)
    print("MAKİNE ÖĞRENMESİ MODEL DEĞERLENDİRMESİ")
    print("=" * 60)

    # Golub veri setinin orijinal eğitim-test ayrımını koru.
    # Eğitim ve bağımsız test dosyaları load_golub()
    # içerisinde bu sırayla birleştirilmiştir.

    n_train = metadata["train_samples"]

    X_train = X.iloc[:n_train].copy()
    X_test = X.iloc[n_train:].copy()

    y_train = np.asarray(y[:n_train])
    y_test = np.asarray(y[n_train:])

    # Eksik etiketli örnekleri çıkar.
    train_valid = pd.notna(y_train)
    test_valid = pd.notna(y_test)

    X_train = X_train.iloc[train_valid]
    X_test = X_test.iloc[test_valid]

    y_train = y_train[train_valid].astype(int)
    y_test = y_test[test_valid].astype(int)

    print("Eğitim örnekleri:", len(X_train))
    print("Test örnekleri:", len(X_test))
    print("Eğitim gen sayısı:", X_train.shape[1])

    # Eğitimde tamamen eksik olan genleri kaldır.
    valid_genes = X_train.notna().any(axis=0)

    X_train = X_train.loc[:, valid_genes]
    X_test = X_test.loc[:, valid_genes]

    # Sonsuz değerleri eksik değer olarak ele al.
    X_train = X_train.replace(
        [np.inf, -np.inf], np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf], np.nan
    )

    # Modeller
    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED_VALUE,
            n_jobs=-1
        ),

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED_VALUE
        ),

        "SVM": SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=SEED_VALUE
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=SEED_VALUE
        )
    }

    results = []

    for model_name, classifier in models.items():

        print("\n" + "-" * 50)
        print("MODEL:", model_name)
        print("-" * 50)

        # Bütün ön işleme adımları yalnızca
        # eğitim verisi üzerinde öğrenilir.
        pipeline = Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            # Welch yerine ANOVA F-testi ile
            # en ayırt edici 200 gen seçilir.
            ("feature_selection", SelectKBest(
                score_func=f_classif,
                k=min(200, X_train.shape[1])
            )),

            ("scaler", StandardScaler()),

            ("classifier", classifier)
        ])

        # Model eğitimi
        pipeline.fit(X_train, y_train)

        # Test verisi üzerinde tahmin
        y_pred = pipeline.predict(X_test)

        # Temel metrikler
        accuracy = accuracy_score(
            y_test, y_pred
        )

        precision = precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        f1_macro = f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        f1_weighted = f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        # ROC-AUC
        try:
            probabilities = pipeline.predict_proba(
                X_test
            )

            auc = roc_auc_score(
                y_test,
                probabilities[:, 1]
            )

        except (ValueError, IndexError):
            auc = np.nan

        # Sonuçları sakla
        result = {
            "Dataset": "Golub",
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision_Macro": precision,
            "Recall_Macro": recall,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "ROC_AUC": auc,
            "Selected_Genes": min(
                200, X_train.shape[1]
            )
        }

        results.append(result)

        print("Accuracy:", round(accuracy, 4))
        print("Precision (Macro):", round(precision, 4))
        print("Recall (Macro):", round(recall, 4))
        print("F1 (Macro):", round(f1_macro, 4))
        print("F1 (Weighted):", round(f1_weighted, 4))
        print("ROC-AUC:", round(auc, 4))

        print("\nClassification Report:")

        print(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["ALL", "AML"],
            zero_division=0
        ))

        # Confusion Matrix
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["ALL", "AML"],
            yticklabels=["ALL", "AML"]
        )

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/Golub_{model_name.replace(' ', '_')}_confusion_matrix.png",
            dpi=300
        )

        plt.close()

    # Bütün modellerin karşılaştırılması
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"{OUTPUT_DIR}/model_performance.csv",
        index=False
    )

    print("\nMODEL KARŞILAŞTIRMA TABLOSU")

    print(
        results_df.to_string(index=False)
    )

    print("\nModel sonuçları kaydedildi:")
    print(
        f"{OUTPUT_DIR}/model_performance.csv"
    )

    return results_df

def compare_datasets(datasets):

    print("\nORTAK GEN KARŞILAŞTIRMASI")

    names = list(datasets.keys())

    common = set(
        datasets[names[0]][0].columns
    )

    for name in names[1:]:

        common = common.intersection(
            datasets[name][0].columns
        )

    print(
        "Tüm veri setlerinde ortak gen:",
        len(common)
    )

    pd.DataFrame({
        "Dataset": names,
        "Gene_Count": [
            datasets[n][0].shape[1]
            for n in names
        ],
        "Sample_Count": [
            datasets[n][0].shape[0]
            for n in names
        ],
        "Common_Genes": [
            len(common)
            for n in names
        ]
    }).to_csv(
        f"{OUTPUT_DIR}/dataset_comparison.csv",
        index=False
    )

    # Ortak genler üzerinden örnek bazında
    # ifade dağılımlarını karşılaştır.
    if len(common) > 0:

        common = sorted(common)

        fig, axes = plt.subplots(
            1, len(names),
            figsize=(6 * len(names), 5)
        )

        if len(names) == 1:
            axes = [axes]

        for ax, name in zip(axes, names):

            X = datasets[name][0][common]

            values = X.values.flatten()

            values = values[
                np.isfinite(values)
            ]

            sns.histplot(
                values,
                bins=60,
                stat="density",
                ax=ax
            )

            ax.set_title(name)
            ax.set_xlabel("Expression")

        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/dataset_expression_comparison.png",
            dpi=300
        )

        plt.close()

def main():

    print("=" * 60)
    print("EDA ANALİZİ BAŞLIYOR")
    print("=" * 60)

    datasets = load_all()

    quality_reports = []

    golub_X, golub_y, golub_metadata = datasets["Golub"]

    model_results = train_and_evaluate_models(
        golub_X,
        golub_y,
        golub_metadata
    )

    print(model_results)

    for name, (X, y, metadata) in datasets.items():

        print("\n", "=" * 60)
        print(name, "ANALİZ EDİLİYOR")
        print("=" * 60)

        print(
            "Normalizasyon:",
            metadata["normalization"]
        )

        # Kalite raporu
        quality = data_quality(
            X, y, name
        )

        quality["Normalization"] = (
            metadata["normalization"]
        )

        quality_reports.append(quality)

        # Analizler
        descriptive_statistics(
            X, name
        )

        expression_distribution(
            X, name
        )

        class_distribution(
            y, name
        )

        dimensionality_reduction(
            X, y, name
        )

        genes = differential_gene_analysis(
            X, y, name
        )

        create_heatmap(
            X, y, genes, name
        )

        print(
            f"{name} analizleri tamamlandı."
        )

    # Veri setleri arası karşılaştırma
    compare_datasets(datasets)

    pd.DataFrame(
        quality_reports
    ).to_csv(
        f"{OUTPUT_DIR}/data_quality_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("ARA RAPOR ANALİZLERİ TAMAMLANDI")
    print("Çıktı klasörü:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()