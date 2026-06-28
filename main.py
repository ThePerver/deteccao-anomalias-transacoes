#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de exemplo: Detecção de Anomalias em Transações
Execute com: python main.py
"""

import pandas as pd
import numpy as np
from deteccao_anomalias import DetectorAnomalias, ComparadorAlgoritmos
from sklearn.model_selection import train_test_split


def exemplo_1_isolation_forest():
    """Exemplo 1: Detecção com Isolation Forest (não supervisionado)"""
    print("\n" + "="*70)
    print("EXEMPLO 1: Isolation Forest (Não Supervisionado)")
    print("="*70)
    print("Cenário: Detectar fraudes sem ter dados rotulados\n")
    
    # Carregar dados
    df = pd.read_csv('exemplo_transacoes.csv')
    X = df[['valor', 'hora_dia', 'dias_no_cartao', 'quantidade_transacoes_dia']].values
    
    # Criar detector
    detector = DetectorAnomalias(algoritmo='isolation_forest', contamination=0.1)
    detector.treinar(X)
    
    # Prever (pega anomalias no mesmo conjunto, só pra demo)
    predicoes, scores = detector.prever(X, retornar_score=True)
    
    # Mostrar resultados
    fraudes_detectadas = sum(predicoes == -1)
    print(f"\n✓ Total de transações: {len(predicoes)}")
    print(f"⚠️  Anomalias detectadas: {fraudes_detectadas}")
    print(f"\nTop 5 transações mais suspeitas:")
    
    # Indices das mais suspeitas
    indices_suspeitos = np.argsort(scores)[:5]
    for idx in indices_suspeitos:
        print(f"   - ID {idx}: Valor R${df.iloc[idx]['valor']:.2f}, "
              f"Hora {int(df.iloc[idx]['hora_dia'])}h, "
              f"Score: {scores[idx]:.2f}")


def exemplo_2_random_forest():
    """Exemplo 2: Detecção com Random Forest (supervisionado)"""
    print("\n" + "="*70)
    print("EXEMPLO 2: Random Forest (Supervisionado)")
    print("="*70)
    print("Cenário: Treinar com dados históricos rotulados\n")
    
    # Carregar dados
    df = pd.read_csv('exemplo_transacoes.csv')
    X = df[['valor', 'hora_dia', 'dias_no_cartao', 'quantidade_transacoes_dia']].values
    y = df['anomalia'].values
    
    # Split treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Criar e treinar
    detector = DetectorAnomalias(algoritmo='random_forest')
    detector.treinar(X_treino, y_treino)
    
    # Avaliar
    detector.avaliar(X_teste, y_teste)
    
    # Feature importance
    print("\n📊 Importância das Features:")
    features = ['Valor', 'Hora do Dia', 'Dias no Cartão', 'Qtd Transações']
    importances = detector.feature_importance()
    for feature, importance in zip(features, importances):
        print(f"   {feature:.<30} {importance:.2%}")


def exemplo_3_nova_transacao():
    """Exemplo 3: Testar nova transação"""
    print("\n" + "="*70)
    print("EXEMPLO 3: Testando Nova Transação")
    print("="*70)
    print("Cenário: Uma nova transação chega e precisamos classificá-la\n")
    
    # Carregar dados
    df = pd.read_csv('exemplo_transacoes.csv')
    X = df[['valor', 'hora_dia', 'dias_no_cartao', 'quantidade_transacoes_dia']].values
    
    # Treinar com Isolation Forest
    detector = DetectorAnomalias(algoritmo='isolation_forest')
    detector.treinar(X)
    
    # Nova transação (normal)
    print("📌 Transação 1 (Normal):")
    transacao_normal = np.array([[85.50, 11, 30, 2]])
    resultado, score = detector.prever(transacao_normal, retornar_score=True)
    status = "✓ AUTORIZADA" if resultado[0] == 1 else "❌ BLOQUEADA"
    print(f"   Valor: R$ {transacao_normal[0, 0]:.2f}")
    print(f"   Hora: {int(transacao_normal[0, 1])}h")
    print(f"   Status: {status} (Score: {score[0]:.2f})")
    
    # Nova transação (suspeita)
    print("\n📌 Transação 2 (Suspeita):")
    transacao_suspeita = np.array([[5000.00, 3, 30, 1]])
    resultado, score = detector.prever(transacao_suspeita, retornar_score=True)
    status = "✓ AUTORIZADA" if resultado[0] == 1 else "❌ BLOQUEADA"
    print(f"   Valor: R$ {transacao_suspeita[0, 0]:.2f}")
    print(f"   Hora: {int(transacao_suspeita[0, 1])}h")
    print(f"   Status: {status} (Score: {score[0]:.2f})")
    print(f"   Motivo: Valor extremamente alto em horário incomum")


def exemplo_4_comparacao():
    """Exemplo 4: Comparar múltiplos algoritmos"""
    print("\n" + "="*70)
    print("EXEMPLO 4: Comparação de Algoritmos")
    print("="*70)
    print("Cenário: Qual algoritmo performa melhor?\n")
    
    # Carregar dados
    df = pd.read_csv('exemplo_transacoes.csv')
    X = df[['valor', 'hora_dia', 'dias_no_cartao', 'quantidade_transacoes_dia']].values
    y = df['anomalia'].values
    
    # Split
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Comparar
    comparador = ComparadorAlgoritmos()
    comparador.comparar(
        X_treino, y_treino, X_teste, y_teste,
        algoritmos=['isolation_forest', 'lof', 'random_forest']
    )


def main():
    """Função principal"""
    print("\n" + "🚀 " * 20)
    print("SISTEMA DE DETECÇÃO DE ANOMALIAS EM TRANSAÇÕES")
    print("🚀 " * 20)
    
    while True:
        print("\n\nEscolha um exemplo para executar:")
        print("1. Isolation Forest (Não Supervisionado)")
        print("2. Random Forest (Supervisionado)")
        print("3. Testar Nova Transação")
        print("4. Comparar Múltiplos Algoritmos")
        print("5. Executar Todos os Exemplos")
        print("0. Sair")
        
        opcao = input("\nDigite a opção (0-5): ").strip()
        
        try:
            if opcao == '1':
                exemplo_1_isolation_forest()
            elif opcao == '2':
                exemplo_2_random_forest()
            elif opcao == '3':
                exemplo_3_nova_transacao()
            elif opcao == '4':
                exemplo_4_comparacao()
            elif opcao == '5':
                exemplo_1_isolation_forest()
                exemplo_2_random_forest()
                exemplo_3_nova_transacao()
                exemplo_4_comparacao()
            elif opcao == '0':
                print("\nAté logo! 👋")
                break
            else:
                print("❌ Opção inválida!")
        
        except FileNotFoundError:
            print("❌ Erro: Arquivo 'exemplo_transacoes.csv' não encontrado!")
            print("Certifique-se de que o arquivo está no mesmo diretório.")
        except Exception as e:
            print(f"❌ Erro ao executar: {e}")
        
        input("\n[Pressione ENTER para continuar...]")


if __name__ == "__main__":
    main()
