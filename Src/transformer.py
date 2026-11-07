import gzip
import pandas as pd
import os

def process_series_matrix_to_golub_format(file_path, geo_id):
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı!")
        return

    print(f"--- Processing Local {geo_id} Series Matrix File Directly to Golub Format ---")
    
    # 1. Metadatayı (!) ifadelere göre ayır
    metadata_lines = []
    matrix_start_row = 0
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.startswith('!Sample_characteristics_ch1') or line.startswith('!Sample_title'):
                metadata_lines.append(line.strip())
            if line.startswith('!series_matrix_table_begin'):
                matrix_start_row = idx + 1
                break

    # 2. Gen İfade Matrisini Oku
    print("Reading expression data matrix...")
    expression_df = pd.read_csv(
        file_path, 
        sep='\t', 
        skiprows=matrix_start_row, 
        comment='!', 
        index_col=0,
        compression='gzip'
    )
    
    if '!series_matrix_table_end' in expression_df.index:
        expression_df = expression_df.drop('!series_matrix_table_end')

    # 3. Metadatayı Sözlük Yapısında Topla
    print("Parsing clinical metadata tags...")
    sample_ids = expression_df.columns.tolist()
    clinical_df = pd.DataFrame(index=sample_ids)
    clinical_df['combined_text'] = ""
    
    for line in metadata_lines:
        parts = line.split('\t')
        values = parts[1:]
        if len(values) == len(sample_ids):
            for i, val in enumerate(values):
                clinical_df.at[sample_ids[i], 'combined_text'] += " " + str(val).lower()

    # 4. ALL / AML Filtreleme
    clinical_df['combined_text'] = clinical_df['combined_text'].str.strip()
    mask = clinical_df['combined_text'].str.contains('aml|myeloid|all|lymphoblastic|leukemia', na=False)
    filtered_patients = clinical_df[mask].copy()
    
    if len(filtered_patients) == 0:
        print("⚠️ Warning: Text filters yielded 0 samples. Retaining all dataset samples.")
        filtered_patients = clinical_df.copy()

    # 5. DOĞRUDAN GOLUB FORMATINA ÇEVİRME (Metin olarak ALL / AML)
    cancer_labels = []
    for text in filtered_patients['combined_text']:
        if 'all' in text or 'lymphoblastic' in text:
            cancer_labels.append('ALL')
        else:
            cancer_labels.append('AML')
            
    # Golub formatında sütun isimleri: patient, cancer
    golub_actual_df = pd.DataFrame({
        'patient': filtered_patients.index.astype(str),
        'cancer': cancer_labels
    })
    
    # 6. Sadece Seçilen Hastaların Gen İfade Sütunlarını Süz
    valid_gsm_codes = filtered_patients.index
    expression_df_filtered = expression_df[valid_gsm_codes]
    
    # 7. Nihai CSV Dosyalarını Kaydet
    expression_df_filtered.to_csv(f"{geo_id}_expression_matrix.csv")
    golub_actual_df.to_csv(f"{geo_id}_actual.csv", index=False)
    
    print(f"\n✅ {geo_id} conversion successfully completed with pure Pandas!")
    print(f"Matrix Dimension: {expression_df_filtered.shape} (Genes x Samples)")
    print(f"Output files saved: {geo_id}_expression_matrix.csv, {geo_id}_actual.csv\n")

process_series_matrix_to_golub_format("GSE6891_series_matrix.txt.gz", "GSE6891")
process_series_matrix_to_golub_format("GSE26713_series_matrix.txt.gz", "GSE26713")
process_series_matrix_to_golub_format("GSE13159_series_matrix.txt.gz", "GSE13159")