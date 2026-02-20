import pandas as pd

# ==============================================================================
#  1. CONFIGURAÇÃO DOS ARQUIVOS
# ==============================================================================
# Coloque aqui o nome do arquivo antigo que deu erro
arquivo_antigo = 'dados_imperatriz_limpos_unificado(2022_2025).csv' 

# Nome do novo arquivo que será gerado já padronizado
arquivo_novo = 'dados_imperatriz_limpos_unificado(2022_2025).csv'

print(f"⏳ Lendo o arquivo antigo: {arquivo_antigo}...")

# ==============================================================================
#  2. EXTRAÇÃO E LIMPEZA
# ==============================================================================
# Lê o arquivo forçando o Python a descobrir sozinho se é vírgula ou ponto-e-vírgula
try:
    df = pd.read_csv(arquivo_antigo, sep=None, engine='python')
except FileNotFoundError:
    print(f"❌ ERRO: O arquivo '{arquivo_antigo}' não foi encontrado na pasta.")
    exit()

print(f"🔍 Colunas originais detectadas: {df.columns.tolist()}")

# Remove qualquer espaço em branco antes ou depois do nome da coluna (ex: " DAP " vira "DAP")
df.columns = df.columns.str.strip()

# ==============================================================================
#  3. TRADUÇÃO DE COLUNAS (Caso o antigo tivesse outro nome)
# ==============================================================================
# Se no passado você chamava a coluna de um jeito diferente, coloque aqui.
# Formato: 'Nome Antigo': 'Nome Padrão Novo'
dicionario_de_nomes = {
    'Tensao_Solo_kPa': 'Tensao_Media_Atual_kPa',
    'Chuva_mm': 'Chuva_Acumulada_3d_mm',
    'Tmax_C': 'Tmax_Futura_3d_C'
    # Adicione outras aqui se necessário. Se os nomes já estiverem certos, ele apenas ignora.
}

# Aplica a renomeação
df = df.rename(columns=dicionario_de_nomes)

# ==============================================================================
#  4. VALIDAÇÃO E EXPORTAÇÃO
# ==============================================================================
colunas_obrigatorias = ['Tensao_Media_Atual_kPa', 'Chuva_Acumulada_3d_mm', 'Tmax_Futura_3d_C', 'DAP', 'Irrigacao_mm']

colunas_faltando = [col for col in colunas_obrigatorias if col not in df.columns]

if colunas_faltando:
    print(f"\n❌ ERRO CRÍTICO: Mesmo após a limpeza, as seguintes colunas não foram encontradas:")
    print(colunas_faltando)
    print("Dica: Olhe a lista de colunas originais acima e adicione no 'dicionario_de_nomes' do código.")
else:
    # Salva o arquivo final cravado no padrão internacional (separado por vírgula pura)
    df.to_csv(arquivo_novo, index=False, sep=',')
    print(f"\n✅ SUCESSO! O arquivo foi padronizado e salvo como: {arquivo_novo}")
    print("Agora você pode usar esse arquivo novo no seu modelo ALMMo-0 sem dar erro!")