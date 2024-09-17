import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Novo modelo de dados
dados = {
    "Algoritmo": ["SQLQuery", "SQLQuery", "SQLQuery", "NoSQLQuery", "NoSQLQuery", "NoSQLQuery",
                  "CacheManagement", "CacheManagement", "CacheManagement", "QueryOptimization", 
                  "QueryOptimization", "QueryOptimization"],
    "Complexidade": ["O(n)", "O(n)", "O(n)", "O(log n)", "O(log n)", "O(log n)", 
                      "O(n^2)", "O(n^2)", "O(n^2)", "O(1)", "O(1)", "O(1)"],
    "QuantidadeDados": [1000, 5000, 10000, 1000, 5000, 10000, 1000, 5000, 10000, 1000, 5000, 10000],
    "TempoResposta(ms)": [15, 70, 180, 12, 45, 130, 25, 80, 200, 10, 40, 110],
    "UsoCPU(%)": [30, 55, 75, 25, 50, 70, 40, 65, 80, 20, 45, 60],
    "UsoMemoria(MB)": [40, 180, 360, 35, 160, 320, 50, 200, 400, 30, 150, 300],
    "TaxaErro": [0, 2, 5, 0, 1, 4, 1, 3, 6, 0, 2, 5]
}

df = pd.DataFrame(dados)

# Estatísticas descritivas
estatisticas = df.describe()
print("Estatísticas descritivas:\n", estatisticas)

# Análise de Eficiência: Tempo de Resposta por Quantidade de Dados
plt.figure(figsize=(10, 6))
sns.lineplot(x='QuantidadeDados', y='TempoResposta(ms)', hue='Algoritmo', marker='o', data=df)
plt.title('Tempo de Resposta por Quantidade de Dados')
plt.xlabel('Quantidade de Dados')
plt.ylabel('Tempo de Resposta (ms)')
plt.legend(title='Algoritmo')
plt.show()

# Análise de Escalabilidade: Uso de CPU e Memória com Aumento da Quantidade de Dados
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Gráfico 1: Uso de CPU por Quantidade de Dados
sns.lineplot(ax=axes[0], x='QuantidadeDados', y='UsoCPU(%)', hue='Algoritmo', marker='o', data=df)
axes[0].set_title('Uso de CPU por Quantidade de Dados')
axes[0].set_xlabel('Quantidade de Dados')
axes[0].set_ylabel('Uso de CPU (%)')
axes[0].legend(title='Algoritmo')

# Gráfico 2: Uso de Memória por Quantidade de Dados
sns.lineplot(ax=axes[1], x='QuantidadeDados', y='UsoMemoria(MB)', hue='Algoritmo', marker='o', data=df)
axes[1].set_title('Uso de Memória por Quantidade de Dados')
axes[1].set_xlabel('Quantidade de Dados')
axes[1].set_ylabel('Uso de Memória (MB)')
axes[1].legend(title='Algoritmo')

plt.tight_layout()
plt.show()

# Regressão Linear (Previsão de Tempo de Resposta)

# Definir variáveis independentes e dependentes
X = df[['QuantidadeDados']].values
y = df['TempoResposta(ms)'].values

# Criar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Fazer previsões
y_pred = model.predict(X)

# Gráfico de Previsão Linear
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Previsão linear')
plt.title('Previsão de Tempo de Resposta com Base na Quantidade de Dados')
plt.xlabel('Quantidade de Dados')
plt.ylabel('Tempo de Resposta (ms)')
plt.legend()
plt.show()

# Análise de Confiabilidade: Taxa de Erro por Algoritmo
plt.figure(figsize=(10, 6))
sns.boxplot(x='Algoritmo', y='TaxaErro', data=df)
plt.title('Distribuição da Taxa de Erro por Algoritmo')
plt.xlabel('Algoritmo')
plt.ylabel('Taxa de Erro')
plt.show()