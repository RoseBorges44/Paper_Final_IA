
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importações para os modelos
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)

# Importações para SHAP e LIME
import shap
import lime
import lime.lime_tabular

# Importações para processamento de características temporais
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def carregar_dados_temporais():
    """Carrega os dados com componente temporal"""
    print("Carregando dados com timestamps...")

    # Simulando dados temporais do 3W dataset
    np.random.seed(42)
    n_samples = 2000
    n_features = 10

    # Criar timestamps (dados de 30 dias)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(hours=i*0.36) for i in range(n_samples)]

    # Gerando dados sintéticos com padrão temporal
    X = np.random.randn(n_samples, n_features)

    # Criar anomalias em períodos específicos (simulando falhas intermitentes)
    anomaly_periods = [
        (500, 550),   # Período 1
        (1000, 1050), # Período 2
        (1500, 1530), # Período 3
        (1800, 1850)  # Período 4
    ]

    y = np.zeros(n_samples)
    for start, end in anomaly_periods:
        X[start:end] += np.random.randn(end-start, n_features) * 3
        y[start:end] = 1

    # Adicionar algumas anomalias isoladas
    random_anomalies = np.random.choice(n_samples, 50, replace=False)
    X[random_anomalies] += np.random.randn(50, n_features) * 2.5
    y[random_anomalies] = 1

    # Criar DataFrame
    feature_names = [f'sensor_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['anomaly'] = y
    df['timestamp'] = timestamps

    return df, feature_names

def analisar_isolation_forest_lime(modelo, X_train, X_test, feature_names, n_samples=100):
    """Analisa Isolation Forest usando LIME"""
    print("\nAnalisando Isolation Forest com LIME...")

    # Criar explainer LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Normal', 'Anomalia'],
        mode='classification',
        discretize_continuous=True
    )

    # Função para converter scores do IF em probabilidades
    def if_predict_proba(X):
        scores = modelo.score_samples(X)
        # Normalizar scores para probabilidades
        probs_anomaly = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        probs_normal = 1 - probs_anomaly
        return np.column_stack([probs_normal, probs_anomaly])

    # Analisar algumas instâncias
    lime_importances = []

    # Selecionar amostras: metade anomalias, metade normais
    anomaly_indices = np.where(modelo.predict(X_test) == -1)[0]
    normal_indices = np.where(modelo.predict(X_test) == 1)[0]

    n_each = min(n_samples // 2, len(anomaly_indices), len(normal_indices))
    selected_indices = np.concatenate([
        np.random.choice(anomaly_indices, n_each, replace=False),
        np.random.choice(normal_indices, n_each, replace=False)
    ])

    for idx in selected_indices:
        exp = explainer.explain_instance(
            X_test[idx], 
            if_predict_proba, 
            num_features=len(feature_names)
        )

        # Extrair importâncias
        importance_dict = dict(exp.as_list())
        importances = []
        for feat in feature_names:
            # Procurar a feature nas explicações
            feat_importance = 0
            for feat_exp, imp in importance_dict.items():
                if feat in feat_exp:
                    feat_importance = abs(imp)
                    break
            importances.append(feat_importance)

        lime_importances.append(importances)

    return np.array(lime_importances)

def analisar_anomaly_scores(modelo, X, feature_names):
    """Analisa a contribuição de cada feature para o anomaly score"""
    print("\nAnalisando contribuições para anomaly scores...")

    # Calcular scores base
    base_scores = modelo.score_samples(X)

    # Análise de sensibilidade: perturbar cada feature
    feature_contributions = []

    for i in range(X.shape[1]):
        X_perturbed = X.copy()
        # Perturbar feature i com ruído
        X_perturbed[:, i] += np.random.randn(X.shape[0]) * 0.5

        # Calcular novos scores
        perturbed_scores = modelo.score_samples(X_perturbed)

        # Contribuição = mudança média no score
        contribution = np.abs(base_scores - perturbed_scores).mean()
        feature_contributions.append(contribution)

    return np.array(feature_contributions)

def analisar_temporal_anomalias(df, resultados, nome_modelo):
    """Analisa a distribuição temporal das anomalias"""
    print(f"\nAnalisando distribuição temporal - {nome_modelo}...")

    # Adicionar predições ao dataframe
    df_temp = df.copy()
    df_temp['pred'] = resultados['y_pred']
    df_temp['score'] = resultados['y_proba']

    # Agrupar por hora
    df_temp['hora'] = df_temp['timestamp'].dt.hour
    df_temp['dia'] = df_temp['timestamp'].dt.date

    return df_temp

def visualizar_analise_temporal(todos_resultados_temporais, df_original):
    """Cria visualizações da análise temporal"""
    print("\nGerando visualizações temporais...")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    for i, (modelo, df_temp) in enumerate(todos_resultados_temporais.items()):
        ax = axes[i]

        # Plot temporal das anomalias
        ax.scatter(df_temp[df_temp['anomaly']==0]['timestamp'], 
                  df_temp[df_temp['anomaly']==0]['score'], 
                  alpha=0.5, s=10, label='Normal (Real)')
        ax.scatter(df_temp[df_temp['anomaly']==1]['timestamp'], 
                  df_temp[df_temp['anomaly']==1]['score'], 
                  alpha=0.8, s=20, c='red', label='Anomalia (Real)')

        # Marcar detecções
        deteccoes = df_temp[df_temp['pred']==1]
        ax.scatter(deteccoes['timestamp'], deteccoes['score'], 
                  marker='x', s=50, c='green', label='Detectado')

        ax.set_xlabel('Tempo')
        ax.set_ylabel('Score de Anomalia')
        ax.set_title(f'Detecção Temporal - {modelo}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analise_temporal_anomalias.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Análise de densidade temporal
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (modelo, df_temp) in enumerate(todos_resultados_temporais.items()):
        ax = axes[i]

        # Taxa de detecção por hora do dia
        deteccoes_hora = df_temp[df_temp['pred']==1].groupby('hora').size()
        anomalias_reais_hora = df_temp[df_temp['anomaly']==1].groupby('hora').size()

        horas = range(24)
        det_counts = [deteccoes_hora.get(h, 0) for h in horas]
        real_counts = [anomalias_reais_hora.get(h, 0) for h in horas]

        x = np.arange(len(horas))
        width = 0.35

        ax.bar(x - width/2, real_counts, width, label='Anomalias Reais', alpha=0.8)
        ax.bar(x + width/2, det_counts, width, label='Detecções', alpha=0.8)

        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('Contagem')
        ax.set_title(f'Distribuição por Hora - {modelo}')
        ax.set_xticks(x)
        ax.set_xticklabels(horas)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('distribuicao_horaria_anomalias.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualizar_lime_if(lime_importances, feature_names):
    """Visualiza as explicações LIME para Isolation Forest"""
    print("\nVisualizando explicações LIME...")

    plt.figure(figsize=(12, 8))

    # Calcular importância média e desvio
    mean_importance = lime_importances.mean(axis=0)
    std_importance = lime_importances.std(axis=0)

    # Ordenar por importância
    indices = np.argsort(mean_importance)[::-1]

    # Plot
    x_pos = np.arange(len(feature_names))
    plt.bar(x_pos, mean_importance[indices], yerr=std_importance[indices], 
            capsize=5, alpha=0.7, color='purple')

    plt.xticks(x_pos, [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importância LIME Média')
    plt.title('Explicações LIME para Isolation Forest')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('lime_isolation_forest.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Comparar com análise de sensibilidade
    plt.figure(figsize=(10, 6))

    # Calcular frequência no top-5
    top_5_counts = np.zeros(len(feature_names))
    for imp in lime_importances:
        top_5_idx = np.argsort(imp)[-5:]
        top_5_counts[top_5_idx] += 1

    top_5_freq = (top_5_counts / len(lime_importances)) * 100

    # Ordenar por frequência
    indices = np.argsort(top_5_freq)[::-1]

    plt.bar(range(len(feature_names)), top_5_freq[indices], color='darkviolet', alpha=0.7)
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Frequência no Top-5 (%)')
    plt.title('Estabilidade das Explicações LIME - Isolation Forest')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('estabilidade_lime_isolation_forest.png', dpi=300, bbox_inches='tight')
    plt.close()

def executar_experimento_completo():
    """Executa o experimento completo com análises estendidas"""
    print("Iniciando experimento estendido...")
    print("="*70)

    # Carregar dados temporais
    df, feature_names = carregar_dados_temporais()

    # Separar features e target
    X = df[feature_names].values
    y = df['anomaly'].values
    timestamps = df['timestamp']

    # Split temporal (70% treino, 30% teste)
    split_idx = int(0.7 * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    timestamps_test = timestamps[split_idx:]

    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar modelos
    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
    }

    resultados = {}
    todos_resultados_temporais = {}

    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome}...")

        if nome == 'Isolation Forest':
            modelo.fit(X_train_scaled)
            y_pred = modelo.predict(X_test_scaled)
            y_pred = (y_pred == -1).astype(int)

            scores = modelo.score_samples(X_test_scaled)
            y_proba = 1 - (scores - scores.min()) / (scores.max() - scores.min())

            # Análise LIME
            lime_importances = analisar_isolation_forest_lime(
                modelo, X_train_scaled, X_test_scaled, feature_names
            )

            # Análise de sensibilidade
            sensitivity_scores = analisar_anomaly_scores(
                modelo, X_test_scaled, feature_names
            )

            resultados[nome] = {
                'modelo': modelo,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'lime_importances': lime_importances,
                'sensitivity_scores': sensitivity_scores
            }

        else:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

            resultados[nome] = {
                'modelo': modelo,
                'y_pred': y_pred,
                'y_proba': y_proba
            }

        # Análise temporal
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test['anomaly'] = y_test
        df_test['timestamp'] = timestamps_test.values

        df_temporal = analisar_temporal_anomalias(df_test, resultados[nome], nome)
        todos_resultados_temporais[nome] = df_temporal

    # Visualizações
    visualizar_analise_temporal(todos_resultados_temporais, df)

    # Visualizar LIME para IF
    if 'lime_importances' in resultados['Isolation Forest']:
        visualizar_lime_if(
            resultados['Isolation Forest']['lime_importances'],
            feature_names
        )

    # Comparar técnicas de explicação para IF
    if 'sensitivity_scores' in resultados['Isolation Forest']:
        plt.figure(figsize=(12, 6))

        lime_mean = resultados['Isolation Forest']['lime_importances'].mean(axis=0)
        sensitivity = resultados['Isolation Forest']['sensitivity_scores']

        x = np.arange(len(feature_names))
        width = 0.35

        # Normalizar para comparação
        lime_norm = lime_mean / lime_mean.max()
        sens_norm = sensitivity / sensitivity.max()

        plt.bar(x - width/2, lime_norm, width, label='LIME', alpha=0.8)
        plt.bar(x + width/2, sens_norm, width, label='Análise de Sensibilidade', alpha=0.8)

        plt.xlabel('Features')
        plt.ylabel('Importância Normalizada')
        plt.title('Comparação de Métodos de Explicação - Isolation Forest')
        plt.xticks(x, feature_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('comparacao_explicacao_if.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Gerar relatório
    gerar_relatorio_estendido(resultados, todos_resultados_temporais, y_test)

    print("\nExperimento concluído!")
    print("\nArquivos gerados:")
    print("- analise_temporal_anomalias.png")
    print("- distribuicao_horaria_anomalias.png")
    print("- lime_isolation_forest.png")
    print("- estabilidade_lime_isolation_forest.png")
    print("- comparacao_explicacao_if.png")
    print("- relatorio_analise_estendida.txt")

def gerar_relatorio_estendido(resultados, todos_resultados_temporais, y_test):
    """Gera relatório com análise estendida"""
    relatorio = []
    relatorio.append("ANÁLISE ESTENDIDA - DETECÇÃO DE ANOMALIAS\n")
    relatorio.append("="*60 + "\n")

    # Métricas de desempenho
    relatorio.append("\n1. MÉTRICAS DE DESEMPENHO:\n")
    for nome, res in resultados.items():
        acc = accuracy_score(y_test, res['y_pred'])
        prec = precision_score(y_test, res['y_pred'])
        rec = recall_score(y_test, res['y_pred'])
        f1 = f1_score(y_test, res['y_pred'])
        auc = roc_auc_score(y_test, res['y_proba'])

        relatorio.append(f"\n{nome}:")
        relatorio.append(f"  - Accuracy: {acc:.3f}")
        relatorio.append(f"  - Precision: {prec:.3f}")
        relatorio.append(f"  - Recall: {rec:.3f}")
        relatorio.append(f"  - F1-Score: {f1:.3f}")
        relatorio.append(f"  - AUC-ROC: {auc:.3f}")

    # Análise temporal
    relatorio.append("\n\n2. ANÁLISE TEMPORAL:\n")
    for nome, df_temp in todos_resultados_temporais.items():
        total_anomalias = (df_temp['anomaly'] == 1).sum()
        detectadas = ((df_temp['anomaly'] == 1) & (df_temp['pred'] == 1)).sum()
        falsos_positivos = ((df_temp['anomaly'] == 0) & (df_temp['pred'] == 1)).sum()

        relatorio.append(f"\n{nome}:")
        relatorio.append(f"  - Anomalias reais: {total_anomalias}")
        relatorio.append(f"  - Anomalias detectadas corretamente: {detectadas}")
        relatorio.append(f"  - Falsos positivos: {falsos_positivos}")
        relatorio.append(f"  - Taxa de detecção temporal: {detectadas/total_anomalias:.1%}")

    # Análise LIME para IF
    relatorio.append("\n\n3. ANÁLISE DE EXPLICABILIDADE - ISOLATION FOREST:\n")
    if 'lime_importances' in resultados['Isolation Forest']:
        lime_imp = resultados['Isolation Forest']['lime_importances']
        mean_imp = lime_imp.mean(axis=0)
        top_3_idx = np.argsort(mean_imp)[-3:]

        relatorio.append("\nTop 3 features mais importantes (LIME):")
        for idx in reversed(top_3_idx):
            relatorio.append(f"  - sensor_{idx}: {mean_imp[idx]:.3f}")

    # Conclusões
    relatorio.append("\n\n4. CONCLUSÕES:\n")
    relatorio.append("- A análise temporal mostra padrões de ocorrência de anomalias")
    relatorio.append("- LIME fornece explicações interpretáveis para o Isolation Forest")
    relatorio.append("- A estabilidade das explicações varia entre os métodos")
    relatorio.append("- Recomenda-se combinar múltiplas técnicas de explicação")

    # Salvar relatório
    with open('relatorio_analise_estendida.txt', 'w', encoding='utf-8') as f:
        f.writelines(relatorio)

if __name__ == "__main__":
    executar_experimento_completo()
