import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Etapa 1: Leitura do arquivo CSV
df = pd.read_csv('dados.csv')


estatisticas = df.describe()
print("Estatísticas descritivas:\n", estatisticas)

# Inicio da Criação de gráficos

plt.figure(figsize=(10, 6))
sns.barplot(x='TamanhoEntrada', y='TempoExecucao(ms)', hue='Algoritmo', data=df)
plt.title('Comparação de Tempo de Execução por Algoritmo')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Tempo de Execução (ms)')
plt.legend(title='Algoritmo')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='TamanhoEntrada', y='UsoMemoria(MB)', hue='Algoritmo', marker='o', data=df)
plt.title('Uso de Memória por Algoritmo')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Uso de Memória (MB)')
plt.legend(title='Algoritmo')
plt.show()

#  (Regressão Linear)

# 
X = df[['TamanhoEntrada']].values
y = df['TempoExecucao(ms)'].values


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Previsão linear')
plt.title('Previsão de Tempo de Execução com Base no Tamanho de Entrada')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Tempo de Execução (ms)')
plt.legend()
plt.show()
