import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

arquivo = 'dataset_imperatriz_completo_v4.csv' 

try:
    df = pd.read_csv(arquivo)
    print(f"\n✅ Analisando: {arquivo}")
    
    # 1. Mapeamento de mm para Classe (conforme a lógica do seu projeto)
    def mapear_classe(mm):
        if mm <= 0: return 0
        elif mm <= 2.5: return 1
        elif mm <= 4.5: return 2
        elif mm <= 6.5: return 3
        else: return 4

    df['Classe'] = df['Irrigacao_mm'].apply(mapear_classe)

    print("-" * 50)
    # 2. Verificação de Equilíbrio
    print("\n[1] DISTRIBUIÇÃO DE CLASSES (O Coração do Problema):")
    contagem = df['Classe'].value_counts().sort_index()
    porcentagem = (df['Classe'].value_counts(normalize=True) * 100).sort_index()
    
    for cls in range(5):
        qtd = contagem.get(cls, 0)
        perc = porcentagem.get(cls, 0)
        print(f"   ➤ Classe {cls}: {qtd:>4} exemplos ({perc:>5.2f}%)")

    # 3. Verificação de Cobertura de Sensores
    print("\n[2] LIMITES DOS SENSORES (O que a IA conhece):")
    stats = df[['Tensao_Media_Atual_kPa', 'Tmax_Futura_3d_C', 'Chuva_Acumulada_3d_mm']].agg(['min', 'max'])
    print(stats)

    # 4. Gráfico de Dispersão
    print("\n[3] Gerando gráfico 'Mapa de Calor' do conhecimento...")
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(df['Tensao_Media_Atual_kPa'], 
                         df['Tmax_Futura_3d_C'], 
                         c=df['Classe'], 
                         cmap='viridis', 
                         alpha=0.5,
                         edgecolors='w', 
                         linewidth=0.5)
    
    plt.colorbar(scatter, label='Classe de Irrigação (0 a 4)')
    plt.xlabel('Tensão do Solo (kPa)')
    plt.ylabel('Temperatura Máxima (°C)')
    plt.title('Distribuição de Dados de Treinamento (AquaCrop)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    print("👉 Feche a janela do gráfico para finalizar o script.")
    plt.show()

except Exception as e:
    print(f"❌ Erro ao processar: {e}")