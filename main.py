import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Etapa 1: Leitura do arquivo CSV atualizado
dados = {
    "Algoritmo": ["SQLQuery", "SQLQuery", "SQLQuery", "NoSQLQuery", "NoSQLQuery", "NoSQLQuery",
                  "CacheManagement", "CacheManagement", "CacheManagement", "QueryOptimization", 
                  "QueryOptimization", "QueryOptimization"],
    "TamanhoEntrada": [100, 500, 1000, 100, 500, 1000, 100, 500, 1000, 100, 500, 1000],
    "TempoExecucao(ms)": [12, 60, 150, 8, 25, 70, 10, 30, 65, 6, 20, 55],
    "UsoMemoria(MB)": [35, 160, 320, 30, 120, 250, 32, 140, 280, 29, 130, 260],
    "UsuariosSimultaneos": [5, 5, 10, 5, 10, 10, 5, 10, 15, 5, 10, 15],
    "Erros": [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 0, 1]
}

df = pd.DataFrame(dados)

# Estatísticas descritivas
estatisticas = df.describe()
print("Estatísticas descritivas:\n", estatisticas)

# Início da criação de gráficos

# Gráfico 1: Comparação de Tempo de Execução por Algoritmo
plt.figure(figsize=(10, 6))
sns.barplot(x='TamanhoEntrada', y='TempoExecucao(ms)', hue='Algoritmo', data=df)
plt.title('Comparação de Tempo de Execução por Algoritmo')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Tempo de Execução (ms)')
plt.legend(title='Algoritmo')
plt.show()

# Gráfico 2: Uso de Memória por Algoritmo
plt.figure(figsize=(10, 6))
sns.lineplot(x='TamanhoEntrada', y='UsoMemoria(MB)', hue='Algoritmo', marker='o', data=df)
plt.title('Uso de Memória por Algoritmo')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Uso de Memória (MB)')
plt.legend(title='Algoritmo')
plt.show()

# Regressão Linear (Previsão de Tempo de Execução)

# Definir variáveis independentes e dependentes
X = df[['TamanhoEntrada']].values
y = df['TempoExecucao(ms)'].values

# Criar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Fazer previsões
y_pred = model.predict(X)

# Gráfico de Previsão Linear
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Previsão linear')
plt.title('Previsão de Tempo de Execução com Base no Tamanho de Entrada')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Tempo de Execução (ms)')
plt.legend()
plt.show()
