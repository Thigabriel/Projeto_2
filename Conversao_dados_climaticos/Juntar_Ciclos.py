import pandas as pd

# 1. Lê os seus dois arquivos
df1 = pd.read_csv('dataset_imperatriz_ajustado(Ciclo1).csv')
df2 = pd.read_csv('dataset_imperatriz_ajustado(Ciclo2).csv')

# 2. Junta os dois (Data Augmentation Múltiplos Talhões)
df_final = pd.concat([df1, df2], ignore_index=True)

# 3. Remove duplicatas apenas se TODAS as colunas forem idênticas
df_final = df_final.drop_duplicates()

# 4. Salva o seu Super Dataset!
df_final.to_csv('dataset_imperatriz_completo_v4.csv', index=False)
print("Arquivo final salvo com sucesso!")