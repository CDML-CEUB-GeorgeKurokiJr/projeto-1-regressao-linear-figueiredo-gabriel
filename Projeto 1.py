import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. CARREGAR OS DADOS (Agora pegando todas as colunas relevantes)
try:
    df = pd.read_csv('Sydney_Data.csv', header=None)
    
    # O dataset original tem 49 colunas (48 de posição + 1 de energia total)
    # Vamos usar as 300.000 linhas e TODAS as 49 colunas
    df_final = df.iloc[:300000, :49].copy()
    
    print(f"✅ Dados carregados! Colunas utilizadas: {df_final.shape[1]}")
    print("⏳ Treinando modelo de alta precisão (isso pode levar 1 minuto)...")
except FileNotFoundError:
    print("❌ Arquivo não encontrado.")
    exit()

# 2. SEPARAR DADOS (X = 48 colunas, y = última coluna)
X = df_final.iloc[:, :-1] 
y = df_final.iloc[:, -1]

# 3. DIVIDIR TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODELO TUNADO (Mais árvores e usando todos os processadores)
modelo_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_final.fit(X_train, y_train)

# 5. TESTAR
previsoes = modelo_final.predict(X_test)

# 6. RESULTADOS
r2 = r2_score(y_test, previsoes)
mae = mean_absolute_error(y_test, previsoes)

print("-" * 30)
print(f"🏆 RESULTADO FINAL (48 COLUNAS):")
print(f"Precisão (R² Score): {r2:.4f}")
print(f"Erro Médio Absoluto: {mae:.2f}")
print("-" * 30)

# 7. BÔNUS: GRÁFICO DE COMPARAÇÃO
plt.figure(figsize=(10,6))
plt.scatter(y_test[:500], previsoes[:500], alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Real vs. Predição (Random Forest)')
plt.xlabel('Energia Real')
plt.ylabel('Energia Prevista')
plt.show()