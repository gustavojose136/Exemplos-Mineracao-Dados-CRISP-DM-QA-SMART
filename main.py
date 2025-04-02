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

log_print("============================================")
log_print("          Relatório de Mineração de Dados")
log_print("============================================\n")

# ======================================================
# 1. Técnicas de Mineração de Dados (MD)
# ======================================================
log_print("1. Exemplos de Técnicas de Mineração de Dados (MD)")
log_print("   As técnicas de MD são utilizadas para identificar padrões e extrair conhecimento")
log_print("   a partir de grandes volumes de dados. A seguir, exemplos sucintos de técnicas:")
log_print("   - Classificação: Predição de categorias (ex.: identificar espécies no dataset Iris).")
log_print("   - Regressão: Previsão de valores numéricos (ex.: estimar preços ou demandas).")
log_print("   - Clusterização: Agrupamento de dados semelhantes (ex.: segmentação de clientes).")
log_print("   - Associação: Descoberta de relações entre itens (ex.: análise de cestas de compra).")
log_print("   - Detecção de Anomalias: Identificação de outliers (ex.: detectar transações fraudulentas).")
log_print("------------------------------------------------")

# ----- Classificação -----
log_print("----- Classificação -----")
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
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

X_anom = np.r_[X_anom, [[0, 0], [4, 4]]]
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_anom)
anomaly_labels = iso_forest.predict(X_anom)
log_print("Exemplo de Detecção de Anomalias (Isolation Forest):")
log_print("Rótulos (1 = normal, -1 = anomalia) para os 10 primeiros registros: " + str(anomaly_labels[:10]))
log_print("------------------------------------------------\n")

# ======================================================
# 2. Entendimento de Negócio e Questões Analíticas (QA) - Método SMART
# ======================================================
log_print("2. Problema com Questões Analíticas (QA) utilizando o método SMART")

problema = "Previsão de Demanda e Otimização de Recursos para um Restaurante de Comida Orgânica"
log_print("Problema: " + problema)

log_print("Justificativa da escolha dos dados:")
log_print("  - O setor de alimentação orgânica tem ganhado destaque devido à crescente preocupação com saúde e sustentabilidade.")
log_print("  - A demanda por pratos orgânicos é sensível a fatores sazonais, promoções e eventos locais, necessitando de análises precisas.")
log_print("  - Dados de pedidos (data, prato, quantidade, preço) permitem identificar padrões de consumo e otimizar recursos,")
log_print("    reduzindo desperdícios e melhorando a eficiência operacional.")
log_print("------------------------------------------------")

log_print("Questões Analíticas (QA) baseadas no método SMART:")
qas = {
    "QA1": "Específico: Quais pratos apresentam maior variação na demanda em horários de pico e em períodos promocionais?",
    "QA2": "Mensurável: Qual a porcentagem de variação na demanda entre dias úteis e finais de semana nos últimos 6 meses?",
    "QA3": "Atingível: É possível ajustar o número de funcionários em 20% para atender a variação de demanda sem comprometer a qualidade?",
    "QA4": "Relevante e Temporal: Como a sazonalidade (ex.: verão vs. inverno) impacta a demanda e quais ajustes podem ser planejados para os próximos 12 meses?",
    "QA5": "Específico: Quais pratos geram maior margem de lucro e qual sua contribuição para o faturamento total?",
    "QA6": "Mensurável: Qual a taxa de retorno dos clientes após a implementação de campanhas promocionais específicas?",
    "QA7": "Atingível: Quais estratégias podem reduzir o desperdício de alimentos em 15% nos próximos 6 meses?",
    "QA8": "Relevante e Temporal: Qual o impacto das condições climáticas na escolha dos pratos e no volume de pedidos durante o verão?"
}

for chave, questao in qas.items():
    log_print(f"{chave}: {questao}")
log_print("------------------------------------------------\n")

# ======================================================
# 3. Geração de Dados Brutos e Atividades de Entendimento e Preparação (CRISP-DM)
# ======================================================
log_print("3. Geração de Dados Brutos e Atividades de Entendimento e Preparação de Dados (CRISP-DM)")
log_print("Utilizando o domínio de negócio: " + problema)
fake = Faker('pt_BR')

# Simulação de 100 pedidos em um restaurante de comida orgânica
num_pedidos = 100
pratos = ['Salada Orgânica', 'Sopa Detox', 'Quinoa Bowl', 'Smoothie Verde', 'Wrap Vegetariano', 'Suco Natural']
dados_brutos = []

for i in range(1, num_pedidos + 1):
    pedido = {
        "order_id": i,
        "order_date": fake.date_between(start_date='-1y', end_date='today'),
        "prato": random.choice(pratos),
        "quantidade": random.randint(1, 5),
        "preco_unitario": round(random.uniform(15.0, 50.0), 2),
        "table_number": random.randint(1, 20),
        "customer_id": fake.random_int(min=1000, max=5000)
    }
    dados_brutos.append(pedido)

df_raw = pd.DataFrame(dados_brutos)
log_print("Dados Brutos (primeiras 5 linhas):")
log_print(str(df_raw.head()))
log_print("------------------------------------------------")


log_print("Atividades de Entendimento dos Dados:")
info_buf = StringIO()
df_raw.info(buf=info_buf)
log_print(info_buf.getvalue())
log_print("Estatísticas Descritivas:")
log_print(str(df_raw.describe()))

log_print("Contagem de valores duplicados: " + str(df_raw.duplicated().sum()))
log_print("------------------------------------------------")

log_print("Atividades de Preparação dos Dados:")

df_raw['order_date'] = pd.to_datetime(df_raw['order_date'])
log_print("Dados após conversão de 'order_date' para datetime:")
log_print(str(df_raw.head()))

log_print("Contagem de valores nulos por coluna:")
log_print(str(df_raw.isnull().sum()))

df_raw['total_value'] = df_raw['quantidade'] * df_raw['preco_unitario']
log_print("Dados após preparação (nova coluna 'total_value'):")
log_print(str(df_raw.head()))

df_raw = df_raw.drop_duplicates()
log_print("Dados após remoção de duplicatas:")
log_print(str(df_raw.head()))
log_print("------------------------------------------------\n")

log_print("Fim do relatório.")
results_file.close()
