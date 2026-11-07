import pandas as pd

# 1. Actual dosyalarını yükle ve alt alta (satır bazlı) birleştir
gse6891_actual = pd.read_csv("GSE6891_actual.csv")
gse26713_actual = pd.read_csv("GSE26713_actual.csv")

# Satır bazlı birleştirme (axis=0)
gse6891_gse26713_actual = pd.concat([gse6891_actual, gse26713_actual], axis=0, ignore_index=True)
gse6891_gse26713_actual.to_csv("GSE6891_GSE26713_actual.csv", index=False)


# 2. Expression Matrix dosyalarını yükle ve birleştir
gse6891_expr = pd.read_csv("GSE6891_expression_matrix.csv", index_col=0)
gse26713_expr = pd.read_csv("GSE26713_expression_matrix.csv", index_col=0)

# Gen ekspresyon matrisleri genelde ortak genlere (satırlara) göre yan yana (sütun bazlı, axis=1) birleştirilir
gse6891_gse26713_expression_matrix = pd.concat([gse6891_expr, gse26713_expr], axis=1, join="inner")
gse6891_gse26713_expression_matrix.to_csv("GSE6891_GSE26713_expression_matrix.csv")

print("Dosyalar başarıyla birleştirildi ve kaydedildi!")