# Bioconductor paketlerini yükle
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}

BiocManager::install(
    c("hu6800.db", "AnnotationDbi"),
    ask = FALSE,
    update = FALSE
)

library(hu6800.db)
library(AnnotationDbi)

# Hu6800 prob kimliklerini al
probe_ids <- keys(
    hu6800.db,
    keytype = "PROBEID"
)

# Gen sembollerini eşleştir
annotation <- AnnotationDbi::select(
    hu6800.db,
    keys = probe_ids,
    columns = c("SYMBOL"),
    keytype = "PROBEID"
)

# Boş eşleşmeleri temizle
annotation <- annotation[
    !is.na(annotation$SYMBOL) &
    annotation$SYMBOL != "",
]

# Tekrarlanan prob kayıtlarını kaldır
annotation <- annotation[
    !duplicated(annotation$PROBEID),
]

# Python için uygun sütun isimleri
colnames(annotation) <- c(
    "Gene Accession Number",
    "Gene Symbol"
)

# CSV olarak kaydet
write.csv(
    annotation,
    "golub_probe_annotation.csv",
    row.names = FALSE
)

cat("Anotasyon dosyası oluşturuldu!\n")
cat("Eşleşen prob sayısı:", nrow(annotation), "\n")