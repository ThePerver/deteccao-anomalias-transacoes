"""
Detecção de Anomalias em Transações Financeiras
Módulo principal com classes para diferentes algoritmos
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


class DetectorAnomalias:
    """
    Classe principal para detecção de anomalias em transações
    Suporta múltiplos algoritmos (supervisionados e não supervisionados)
    """
    
    def __init__(self, algoritmo='isolation_forest', contamination=0.05):
        """
        Inicializa o detector
        
        Args:
            algoritmo (str): Escolha entre 'isolation_forest', 'lof', 'random_forest', 'svm', 'decision_tree'
            contamination (float): Proporção esperada de anomalias (0-1)
        """
        self.algoritmo = algoritmo
        self.contamination = contamination
        self.modelo = None
        self.scaler = StandardScaler()
        self.dados_treinamento = None
        self.metricas = {}
        
        self._inicializar_modelo()
    
    def _inicializar_modelo(self):
        """Inicializa o modelo escolhido"""
        if self.algoritmo == 'isolation_forest':
            self.modelo = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.algoritmo == 'lof':
            self.modelo = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
        elif self.algoritmo == 'random_forest':
            self.modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
        elif self.algoritmo == 'svm':
            self.modelo = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif self.algoritmo == 'decision_tree':
            self.modelo = DecisionTreeClassifier(
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Algoritmo '{self.algoritmo}' não suportado")
    
    def treinar(self, X_treino, y_treino=None):
        """
        Treina o modelo
        
        Args:
            X_treino: Dados de treinamento (array ou DataFrame)
            y_treino: Labels (opcional, obrigatório para supervisionados)
        """
        # Converter para array se DataFrame
        if isinstance(X_treino, pd.DataFrame):
            X_treino = X_treino.values
        
        # Normalizar dados
        X_normalizado = self.scaler.fit_transform(X_treino)
        self.dados_treinamento = X_normalizado
        
        # Treinar
        if self.algoritmo in ['isolation_forest', 'lof']:
            # Não supervisionados
            self.modelo.fit(X_normalizado)
            print(f"✓ Modelo {self.algoritmo} treinado com sucesso")
        else:
            # Supervisionados
            if y_treino is None:
                raise ValueError("y_treino é obrigatório para algoritmos supervisionados")
            self.modelo.fit(X_normalizado, y_treino)
            print(f"✓ Modelo {self.algoritmo} treinado com sucesso")
    
    def prever(self, X_teste, retornar_score=False):
        """
        Faz previsões em novos dados
        
        Args:
            X_teste: Dados para predição
            retornar_score: Se True, retorna também o score de anomalia
        
        Returns:
            predictions: Array com -1 (anomalia) e 1 (normal)
            scores (opcional): Score de anomalia para cada amostra
        """
        if isinstance(X_teste, pd.DataFrame):
            X_teste = X_teste.values
        
        X_normalizado = self.scaler.transform(X_teste)
        
        if self.algoritmo in ['isolation_forest', 'lof']:
            predicoes = self.modelo.predict(X_normalizado)
            if retornar_score:
                scores = self.modelo.score_samples(X_normalizado)
                return predicoes, scores
            return predicoes
        else:
            predicoes = self.modelo.predict(X_normalizado)
            # Converte para formato consistente (-1 = fraude, 1 = normal)
            predicoes = np.where(predicoes == 0, -1, 1)
            if retornar_score:
                scores = self.modelo.predict_proba(X_normalizado)[:, 1]
                return predicoes, scores
            return predicoes
    
    def avaliar(self, X_teste, y_teste):
        """
        Avalia o modelo em dados de teste
        
        Args:
            X_teste: Dados de teste
            y_teste: Labels verdadeiros
        """
        predicoes = self.prever(X_teste)
        
        # Ajustar labels para formato consistente
        y_teste_ajustado = np.where(y_teste == 0, -1, 1)
        
        # Calcular métricas
        self.metricas = {
            'precisao': precision_score(y_teste_ajustado, predicoes, zero_division=0),
            'recall': recall_score(y_teste_ajustado, predicoes, zero_division=0),
            'f1': f1_score(y_teste_ajustado, predicoes, zero_division=0),
            'auc_roc': roc_auc_score(np.where(y_teste == 0, 0, 1), 
                                     self.prever(X_teste, retornar_score=True)[1]),
        }
        
        print("\n" + "="*50)
        print(f"AVALIAÇÃO - {self.algoritmo.upper()}")
        print("="*50)
        print(f"Precisão:  {self.metricas['precisao']:.2%}")
        print(f"Recall:    {self.metricas['recall']:.2%}")
        print(f"F1-Score:  {self.metricas['f1']:.2%}")
        print(f"AUC-ROC:   {self.metricas['auc_roc']:.2%}")
        print("="*50)
        
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_teste_ajustado, predicoes)
        print(f"VP (Verdadeiro Positivo): {cm[0, 0]}")
        print(f"FP (Falso Positivo):      {cm[0, 1]}")
        print(f"FN (Falso Negativo):      {cm[1, 0]}")
        print(f"VN (Verdadeiro Negativo): {cm[1, 1]}")
    
    def feature_importance(self):
        """Retorna importância das features (se disponível)"""
        if hasattr(self.modelo, 'feature_importances_'):
            return self.modelo.feature_importances_
        else:
            print(f"Algoritmo {self.algoritmo} não fornece feature importance")
            return None


class ComparadorAlgoritmos:
    """
    Compara desempenho de múltiplos algoritmos
    """
    
    def __init__(self):
        self.resultados = {}
        self.algoritmos = ['isolation_forest', 'lof', 'random_forest', 'svm', 'decision_tree']
    
    def comparar(self, X_treino, y_treino, X_teste, y_teste, algoritmos=None):
        """
        Treina e avalia múltiplos algoritmos
        
        Args:
            X_treino, y_treino: Dados de treinamento
            X_teste, y_teste: Dados de teste
            algoritmos: Lista de algoritmos a comparar (None = todos)
        """
        if algoritmos is None:
            algoritmos = self.algoritmos
        
        for algo in algoritmos:
            print(f"\n🔄 Treinando {algo}...")
            detector = DetectorAnomalias(algoritmo=algo)
            
            if algo in ['isolation_forest', 'lof']:
                detector.treinar(X_treino)
            else:
                detector.treinar(X_treino, y_treino)
            
            detector.avaliar(X_teste, y_teste)
            self.resultados[algo] = detector.metricas
        
        return self._gerar_resumo()
    
    def _gerar_resumo(self):
        """Gera resumo comparativo"""
        df_resultados = pd.DataFrame(self.resultados).T
        
        print("\n" + "="*70)
        print("COMPARAÇÃO DE ALGORITMOS")
        print("="*70)
        print(df_resultados.round(4))
        print("="*70)
        
        return df_resultados


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # Gerar dados de exemplo
    np.random.seed(42)
    
    # Dados normais
    dados_normais = np.random.normal(loc=[100, 10], scale=[50, 2], size=(1000, 2))
    
    # Dados anômalos
    dados_anomalos = np.array([
        [500, 2],   # Valor muito alto
        [50, 23],   # Hora estranha
        [5000, 1],  # Valor extremo
        [300, 3],   # Valor alto com hora estranha
    ])
    
    # Combinar
    X = np.vstack([dados_normais, dados_anomalos])
    y = np.hstack([np.zeros(1000), np.ones(4)])
    
    # Split treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Teste 1: Isolation Forest (não supervisionado)
    print("\n🚀 TESTE 1: Isolation Forest (Não Supervisionado)")
    detector_if = DetectorAnomalias(algoritmo='isolation_forest')
    detector_if.treinar(X_treino)
    predicoes = detector_if.prever(X_teste)
    print(f"Anomalias detectadas: {sum(predicoes == -1)}")
    
    # Teste 2: Random Forest (supervisionado)
    print("\n🚀 TESTE 2: Random Forest (Supervisionado)")
    detector_rf = DetectorAnomalias(algoritmo='random_forest')
    detector_rf.treinar(X_treino, y_treino)
    detector_rf.avaliar(X_teste, y_teste)
    
    # Teste 3: Comparação
    print("\n🚀 TESTE 3: Comparação de Algoritmos")
    comparador = ComparadorAlgoritmos()
    comparador.comparar(X_treino, y_treino, X_teste, y_teste)
