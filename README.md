# Paper_Final_IA
Trabalho Final da disciplina de IA, Prof Didier, Unifesp

# Explicabilidade e Robustez em Modelos de Detecção de Anomalias Industriais

## 📊 Visão Geral do Projeto

Este projeto aborda um desafio crítico na aplicação de modelos de Machine Learning (ML) em **ambientes industriais críticos**: o "paradoxo da precisão". Embora modelos supervisionados, como Random Forest e XGBoost, demonstrem alta precisão na detecção de anomalias [1], suas explicações geradas por métodos como SHAP frequentemente apresentam **instabilidade crítica, chegando a 0% em testes de estabilidade** [1, 2]. Essa inconsistência compromete a confiança operacional e a aplicação prática, pois a compreensão das causas-raiz das anomalias é tão vital quanto sua detecção precisa [3, 4].

Propomos um **framework híbrido inovador** que integra a alta precisão preditiva de modelos supervisionados com a estabilidade interpretativa proporcionada por abordagens não supervisionadas, como o Isolation Forest associado ao método LIME [3, 5, 6]. Os resultados experimentais demonstram que essa abordagem não apenas mantém elevados níveis de precisão na detecção de anomalias, mas também fornece **explicações consistentemente estáveis e confiáveis**, essenciais para a tomada de decisões em operações industriais [7].

## 🎯 Contribuições Principais

Este trabalho se destaca pelas seguintes contribuições:

1.  **Avaliação Comparativa de Desempenho**: Estabelecimento de **benchmarks superiores** com modelos supervisionados (Random Forest, XGBoost e MLP), que alcançaram F1-scores acima de 0,98 no 3W Dataset [8-10]. Isso superou significativamente os benchmarks anteriores da literatura, como o F1-score de 0,72 de Vargas (2019) e 0,858 de Fernandes Jr. et al. (2024) [11-14].
2.  **Revelação da Instabilidade Explicativa Crítica**: Demonstração, através de 30 execuções independentes, da **instabilidade severa das explicações SHAP** para modelos supervisionados (Random Forest e XGBoost), com um score de estabilidade de **0,00%** [2, 8, 15, 16]. Essa inconsistência levanta sérias preocupações sobre sua utilidade prática no diagnóstico de causas [4].
3.  **Validação de Explicações Estáveis em Modelos Não Supervisionados**: Aplicação bem-sucedida do **método LIME ao Isolation Forest**, demonstrando sua capacidade de fornecer explicações mais estáveis e fisicamente coerentes [8, 17-20].
4.  **Proposição de um Framework Híbrido**: Desenvolvimento de diretrizes práticas para combinar a alta precisão dos modelos supervisionados (para alertas) com a estabilidade interpretativa dos modelos não supervisionados (para diagnóstico de segunda opinião), oferecendo uma solução robusta e operacionalmente confiável [5, 6, 21].

## 📚 Palavras-chave

Detecção de anomalias, explicabilidade, SHAP, manutenção preditiva, 3W Dataset [7].

## 📊 Dataset Utilizado

A pesquisa utilizou o **3W Dataset**, um **recurso público notável fornecido pela Petrobras** [22-24]. Este conjunto de dados reúne 50.913.215 amostras de medições multivariadas de sensores (pressão, temperatura, vazão) instalados em poços de petróleo offshore [25]. As instâncias são categorizadas em condições normais e oito classes de anomalias reais [22, 26].

O 3W Dataset apresenta desafios significativos:
*   **Ruído e Valores Ausentes**: Existem 4947 variáveis ausentes (31,17%) e 1535 variáveis "congeladas" (9,67%) devido a problemas em sensores ou redes de comunicação [25, 27].
*   **Desbalanceamento de Classes**: Contém 597 instâncias normais e 1397 instâncias anômalas, refletindo o desbalanceamento natural em cenários de falha industrial [28].
*   **Natureza Temporal**: Embora os dados sejam originalmente temporais, este estudo optou por uma abordagem de amostras vetoriais independentes para facilitar a análise da explicabilidade [29].

## 🛠️ Metodologia

### Pré-processamento e Engenharia de Atributos
O pipeline de pré-processamento incluiu [29, 30]:
*   **Descarte de variáveis** com alto número de valores ausentes ou variação insignificante [31].
*   **Preenchimento de valores ausentes** com a mediana dos atributos [31, 32].
*   **Tratamento de outliers extremos** por clipagem (3 vezes o Intervalo Interquartil - IQR) [31, 32].
*   **Extração de estatísticas simples** de cada série temporal (média, desvio padrão, máximo e mínimo) [31].
*   **Criação de novas *features*** como média/desvio padrão de pressões e temperaturas, e razões de pressão e vazão/pressão [33, 34].
*   **Padronização dos dados** com `RobustScaler` (menos sensível a *outliers*) [33, 34].
*   **Balanceamento de classes** usando `class_weight='balanced'` (Random Forest) e `scale_pos_weight` (XGBoost) [33].
*   **Divisão estratificada** em conjuntos de treinamento e teste [33, 35].

### Modelos de Machine Learning Utilizados
Foram empregados os seguintes modelos [36, 37]:
*   **Random Forest**: Modelo de *ensemble* robusto, conhecido pela capacidade de generalização [36].
*   **XGBoost**: Alternativa supervisionada de alta performance [36].
*   **Isolation Forest**: Método não supervisionado, utilizado como linha de base para comparação [36]. Treinado apenas com dados normais [38].
*   **Multilayer Perceptron (MLP)**: Rede neural supervisionada [37].
*   **Autoencoder**: Rede neural simétrica não supervisionada, treinada para reconstruir a classe normal [37, 39].

### Técnicas de Explicabilidade
*   **SHAP (SHapley Additive exPlanations)**: Aplicado aos modelos supervisionados (Random Forest e XGBoost) para analisar a importância dos atributos [40]. A estabilidade foi avaliada com **30 execuções independentes** utilizando diferentes sementes aleatórias [40-42].
*   **LIME (Local Interpretable Model-agnostic Explanations)**: Empregado para gerar explicações locais para o Isolation Forest, superando sua limitação de interpretabilidade [43, 44].

### Métricas de Avaliação
*   **Desempenho Preditivo**: F1-Score (métrica principal para datasets desbalanceados), Precisão, Recall e AUC-ROC [43, 45]. A avaliação foi realizada com validação cruzada estratificada (5-fold) para modelos supervisionados [45, 46].
*   **Robustez Interpretativa**: Medida pela frequência com que os atributos apareceram entre os mais importantes nas 30 execuções SHAP [45].

## 🚀 Resultados e Discussão

### Desempenho Preditivo
Os **modelos supervisionados (Random Forest, XGBoost e MLP) estabeleceram um novo patamar de performance** para o 3W Dataset, com F1-scores consistentemente superiores a 0.98 [9, 10]. O Random Forest obteve o melhor resultado com F1-score de **0.9905** e AUC-ROC de **0.9974** [47, 48].

Nosso Isolation Forest replicado, mesmo sendo não supervisionado e beneficiado por um pré-processamento robusto, atingiu um F1-score de **0.8912**, superando o benchmark de Vargas (2019) [11, 12, 49]. O Autoencoder falhou neste experimento, com F1-score próximo de 0 [49, 50].

### Comparação com Benchmarks da Literatura
Os resultados do presente trabalho superam significativamente os marcos importantes da literatura que utilizaram o mesmo dataset [12]:
*   **Vargas et al. (2019)**: F1-score de 0,727 com Isolation Forest [11-14, 24, 51].
*   **Fernandes Jr. et al. (2024)**: F1-score de 0,858 com LOF [11, 12, 14, 22, 52].

### Instabilidade das Explicações SHAP
A análise de estabilidade das explicações SHAP, realizada em 30 execuções independentes, revelou uma **instabilidade completa** [2, 16]. Tanto o Random Forest quanto o XGBoost apresentaram um **score de estabilidade de 0.00%** [2, 20, 49]. Isso significa que as *features* consideradas mais importantes variaram drasticamente a cada execução [2]. Por exemplo, para o Random Forest, a variável mais frequente apareceu no Top-5 em apenas 10% das execuções [15]. Para o XGBoost, nenhuma variável apareceu mais de uma vez [15].

Essa inconsistência compromete a confiança no sistema de IA, pois um engenheiro de campo pode receber justificativas diferentes para eventos similares [4]. A instabilidade é atribuída à aleatoriedade intrínseca dos modelos de *ensemble*, alta colinearidade entre sensores e granularidade extrema do dataset [4, 16, 53].

### Análise de Interpretabilidade com LIME para o Isolation Forest
Em contraste, a análise com LIME para o Isolation Forest forneceu **insights valiosos e explicações mais estáveis** [17, 18]. As cinco *features* mais importantes, com base no valor médio de importância, foram [17, 20]:
1.  T-TPT (0.007)
2.  mean_temp (0.007)
3.  T-JUS-CKP (0.004)
4.  pressure_ratio (0.003)
5.  P-TPT (0.002)

Isso sugere que, embora o Isolation Forest tenha desempenho preditivo inferior aos supervisionados, ele pode servir como uma **ferramenta indispensável de auditoria e "segundo par de olhos"**, fornecendo um diagnóstico mais ancorado na física dos dados quando as explicações dos modelos supervisionados são voláteis [18, 19].

### Consequências Operacionais e Framework Híbrido
Os resultados confirmam o "paradoxo precisão vs. confiança" [5, 18]. Para mitigar o risco da instabilidade do SHAP, propõe-se uma abordagem híbrida: usar a **alta precisão dos modelos supervisionados para gerar alertas**, e empregar um modelo como o **Isolation Forest, auditado com LIME, para fornecer uma segunda opinião com explicações mais estáveis e coerentes** para o diagnóstico da causa-raiz [5, 6, 54].

## ➡️ Trabalhos Futuros

Este estudo abre diversas avenidas para pesquisas futuras, focando em transformar modelos de alta precisão em ferramentas verdadeiramente confiáveis para manutenção preditiva industrial [54]:

*   **Aprofundamento na Instabilidade Explicativa**: Explorar técnicas como a agregação de valores SHAP de múltiplas execuções para gerar explicações consolidadas e robustas, ou a regularização de modelos para forçar a consistência na seleção de *features* [6, 55].
*   **Arquiteturas Robustas à Colinearidade**: Investigar modelos como Redes Neurais com mecanismos de atenção (*attention mechanisms*) ou *Graph Neural Networks (GNNs)*, que podem aprender a desambiguar contribuições de atributos correlatos [55, 56].
*   **Métricas de Estabilidade Explicativa Mais Granulares**: Desenvolver métricas como o "acordo de ranking de importância" entre múltiplas execuções [56].
*   **Reincorporar a Dimensão Temporal**: Utilizar arquiteturas como Long Short-Term Memory (LSTM) e Transformers para aprimorar a detecção de anomalias dependentes de padrões sequenciais e revelar a dinâmica de importância das variáveis ao longo do tempo [57, 58].
*   **Robustez contra Perturbações Adversariais**: Avaliar como ruídos sutis, variações operacionais inesperadas ou ataques deliberados podem influenciar o desempenho preditivo e a estabilidade das explicações [58, 59].

## 🚀 Como Executar o Projeto

### Requisitos (Bibliotecas Python)
Certifique-se de ter as seguintes bibliotecas instaladas em seu ambiente Python [60-63]:

```bash
numpy
pandas
scikit-learn
xgboost
shap
lime
matplotlib
seaborn
tensorflow # (Para MLP e Autoencoder)

Você pode instalá-las via pip: pip install numpy pandas scikit-learn xgboost shap lime matplotlib seaborn tensorflow
Estrutura do Código
O código principal está estruturado em seções lógicas
:
1. Carregamento e Preparação dos Dados: load_3w_dataset() e preprocess_data()
.
2. Modelos de Machine Learning: create_autoencoder() e train_models() para Random Forest, XGBoost, Isolation Forest, MLP e Autoencoder
.
3. Avaliação e Cross-Validation: evaluate_models()
.
4. Análise de Explicabilidade com SHAP e LIME: analyze_shap_stability() e analyze_lime_isolation_forest()
.
5. Análise Temporal: temporal_analysis()
.
6. Visualizações: create_visualizations() (gera gráficos comparativos, estabilidade SHAP, etc.)
.
7. Geração de Relatório: generate_report() (opcional, pode ser adaptada para markdown)
.
Artefatos e Resultados
Os artefatos de modelo treinados (Random Forest, XGBoost, Isolation Forest, MLP e Autoencoder) estão disponíveis no repositório
.
📝 Informações Adicionais
Para a realização desta pesquisa, foram utilizados agentes de inteligência artificial (ChatGPT, Gemini) para auxiliar em tarefas como codificação, fichamento de informações de melhorias e correção textual e formatação de referências
.
✉️ Contato
• Rosemeri Borges: rose.jbob@gmail.com
📄 Licença
Este projeto é licenciado sob a licença, como o artigo de Fernandes Jr. et al. (2024).
