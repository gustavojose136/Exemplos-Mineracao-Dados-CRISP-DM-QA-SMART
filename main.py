import logging
import numpy as np
import pandas as pd
import random
from io import StringIO
from sklearn.datasets import load_iris, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from faker import Faker

logging.basicConfig(
    filename='execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

results_file = open('resultados.txt', 'w', encoding='utf-8')

def log_print(message):
    print(message)
    results_file.write(message + "\n")
    logging.info(message)

# ================================
# 1. Técnicas de Mineração de Dados (MD)
# ================================
log_print("1. Exemplos de Técnicas de Mineração de Dados (MD)")
log_print("   - Classificação, Regressão, Clusterização, Associação e Detecção de Anomalias.")
log_print("------------------------------------------------")

# ----- Classificação -----
log_print("----- Classificação -----")
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predicoes_classificacao = clf.predict(X_test)
log_print("Exemplo de Classificação (Iris):")
log_print("Predições (primeiras 5): " + str(predicoes_classificacao[:5]))
log_print("------------------------------------------------")

# ----- Regressão -----
log_print("----- Regressão -----")
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)
predicoes_regressao = regressor.predict(X_test_reg)
log_print("Exemplo de Regressão (Dados Sintéticos):")
log_print("Predições (primeiras 5): " + str(predicoes_regressao[:5]))
log_print("------------------------------------------------")

# ----- Clusterização -----
log_print("----- Clusterização -----")
X_blobs, _ = make_blobs(n_samples=100, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_blobs)
log_print("Exemplo de Clusterização (KMeans):")
log_print("Centroides dos clusters:")
log_print(str(kmeans.cluster_centers_))
log_print("------------------------------------------------")

# ----- Associação -----
log_print("----- Associação -----")
transactions = [
    ['leite', 'pão', 'queijo'],
    ['pão', 'manteiga'],
    ['leite', 'pão', 'manteiga', 'queijo'],
    ['leite', 'queijo'],
    ['pão', 'manteiga', 'queijo']
]
log_print("Exemplo de Associação (Market Basket):")
log_print("Transações: " + str(transactions))
log_print("------------------------------------------------")

# ----- Detecção de Anomalias -----
log_print("----- Detecção de Anomalias -----")
rng = np.random.RandomState(42)
X_anom = 0.3 * rng.randn(100, 2)
X_anom = np.r_[X_anom + 2, X_anom - 2]
# Inserindo alguns outliers
X_anom = np.r_[X_anom, [[0, 0], [4, 4]]]
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_anom)
anomaly_labels = iso_forest.predict(X_anom)
log_print("Exemplo de Detecção de Anomalias (Isolation Forest):")
log_print("Rótulos (1 = normal, -1 = anomalia) para os 10 primeiros registros: " + str(anomaly_labels[:10]))
log_print("------------------------------------------------\n")

# ================================
# 2. Problema com Questões Analíticas (QA) utilizando o método SMART
# ================================
log_print("2. Exemplo de Problema com Questões Analíticas (QA) utilizando o método SMART")
problema = "Otimização de Estoque para uma Rede de Lojas de Alimentos Orgânicos"
log_print("Problema: " + problema)
qas = {
    "QA1": "Específico: Quais produtos orgânicos apresentam maior risco de excesso de estoque em cada loja?",
    "QA2": "Mensurável: Qual a porcentagem de produtos com alta vs. baixa rotatividade nos últimos 6 meses?",
    "QA3": "Atingível: É possível reduzir os custos de armazenagem em 15% ajustando os níveis de estoque sem comprometer a disponibilidade?",
    "QA4": "Relevante e Temporal: De que forma a sazonalidade (mensal/trimestral) impacta a demanda e como ajustar os estoques para os próximos 12 meses?"
}
log_print("Questões Analíticas (QA):")
for chave, questao in qas.items():
    log_print(f"{chave}: {questao}")
log_print("------------------------------------------------\n")

# ================================
# 3. Geração de Dados Brutos e Atividades de Entendimento e Preparação de Dados (CRISP-DM)
# ================================
log_print("3. Geração de Dados Brutos e Atividades de Entendimento e Preparação de Dados (CRISP-DM)")
fake = Faker('pt_BR')

# Gerando dados brutos: Simulação de 100 transações em lojas
num_transactions = 100
produtos = ['Maçã Orgânica', 'Banana Orgânica', 'Alface', 'Tomate', 'Cenoura', 'Espinafre']
dados_brutos = []

for _ in range(num_transactions):
    transacao = {
        "store_id": random.choice([1, 2, 3]),  # Supondo 3 lojas
        "transaction_date": fake.date_between(start_date='-1y', end_date='today'),
        "product": random.choice(produtos),
        "quantity": random.randint(1, 20),
        "price": round(random.uniform(2.0, 20.0), 2),
        "customer_id": fake.random_int(min=1000, max=5000)
    }
    dados_brutos.append(transacao)

df_raw = pd.DataFrame(dados_brutos)
log_print("Dados Brutos (primeiras 5 linhas):")
log_print(str(df_raw.head()))
log_print("------------------------------------------------")

log_print("Atividades de Entendimento dos Dados:")
log_print("Informações Gerais dos Dados:")

info_buf = StringIO()
df_raw.info(buf=info_buf)
log_print(info_buf.getvalue())

log_print("Estatísticas Descritivas:")
log_print(str(df_raw.describe()))
log_print("------------------------------------------------")

log_print("Atividades de Preparação dos Dados:")

df_raw['transaction_date'] = pd.to_datetime(df_raw['transaction_date'])
log_print("Dados após conversão de 'transaction_date' para datetime:")
log_print(str(df_raw.head()))

log_print("Contagem de valores nulos por coluna:")
log_print(str(df_raw.isnull().sum()))

df_raw['total_revenue'] = df_raw['quantity'] * df_raw['price']
log_print("\nDados após preparação (exemplo com nova coluna 'total_revenue'):")
log_print(str(df_raw.head()))
log_print("------------------------------------------------")

results_file.close()
