import pandas as pd
import numpy as np

# 1. Carregar o arquivo original
arquivo_orig = 'dataset_imperatriz_completo_v4.csv'
try:
    df = pd.read_csv(arquivo_orig)
    print(f"📚 Dataset Original Carregado: {len(df)} linhas.")
except FileNotFoundError:
    print("❌ Erro: Arquivo não encontrado.")
    exit()

# 2. Mapear mm para Classes (0 a 4)
def mapear_classe(mm):
    if mm <= 0: return 0
    elif mm <= 2.5: return 1
    elif mm <= 4.5: return 2
    elif mm <= 6.5: return 3
    else: return 4

df['Classe'] = df['Irrigacao_mm'].apply(mapear_classe)

# 3. O Pulo do Gato: Balanceamento (Under-sampling)
# Vamos definir um "Teto". Nenhuma classe pode ter mais que X exemplos.
# Como a Classe 3 tem 118 exemplos, vamos usar 130 como teto para equilibrar.
TETO = 130 

df_0 = df[df['Classe'] == 0]
df_1 = df[df['Classe'] == 1]
df_2 = df[df['Classe'] == 2]
df_3 = df[df['Classe'] == 3]
df_4 = df[df['Classe'] == 4]

# Sorteia aleatoriamente (sample) se tiver mais que o Teto
df_0_bal = df_0.sample(n=min(len(df_0), TETO), random_state=42)
df_1_bal = df_1.sample(n=min(len(df_1), TETO), random_state=42) # Classe 1 tinha 189, vai cair pra 130
df_2_bal = df_2.sample(n=min(len(df_2), TETO), random_state=42) # Classe 2 tinha 414, vai cair pra 130
# Classes 3 e 4 mantemos INTEGRAIS (são ouro!)
df_3_bal = df_3 
df_4_bal = df_4 

# 4. Juntar tudo
df_final = pd.concat([df_0_bal, df_1_bal, df_2_bal, df_3_bal, df_4_bal])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True) # Embaralha

# 5. Relatório Final
print("-" * 50)
print("📊 NOVO DATASET BALANCEADO:")
contagem = df_final['Classe'].value_counts().sort_index()
porc = (df_final['Classe'].value_counts(normalize=True) * 100).sort_index()

for cls in range(5):
    qtd = contagem.get(cls, 0)
    pct = porc.get(cls, 0)
    print(f"   ➤ Classe {cls}: {qtd:>3} exemplos ({pct:>5.2f}%)")

print("-" * 50)
print(f"📉 Redução Total: De {len(df)} para {len(df_final)} linhas.")

# 6. Salvar
arquivo_novo = 'dataset_imperatriz_balanceado.csv'
df_final.to_csv(arquivo_novo, index=False)
print(f"✅ Arquivo salvo: {arquivo_novo}")
print("🚀 Agora retreine seu modelo usando ESSE arquivo!")