if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}

BiocManager::install(
    c("hgu133plus2.db", "AnnotationDbi"),
    ask = FALSE,
    update = FALSE
)

library(hgu133plus2.db)
library(AnnotationDbi)

probes <- keys(
    hgu133plus2.db,
    keytype = "PROBEID"
)

annotation <- select(
    hgu133plus2.db,
    keys = probes,
    columns = "SYMBOL",
    keytype = "PROBEID"
)

annotation <- annotation[
    !is.na(annotation$SYMBOL) &
    annotation$SYMBOL != "",
]

annotation <- annotation[
    !duplicated(annotation$PROBEID),
]

colnames(annotation) <- c(
    "Probe_ID",
    "Gene_Symbol"
)

write.csv(
    annotation,
    "gpl570_annotation.csv",
    row.names = FALSE
)

cat("Anotasyon oluşturuldu:", nrow(annotation))