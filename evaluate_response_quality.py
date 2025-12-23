"""
Script pour évaluer la qualité et la justesse des réponses des agents MAGRPO.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json
import re

def evaluate_orchestrator(response: dict, query: str) -> dict:
    """
    Évalue la qualité d'une réponse de l'Orchestrator.
    
    Critères :
    - Format JSON valide
    - Clés présentes
    - Pertinence de la délégation
    - Clarté de l'instruction
    """
    scores = {
        "format_valid": False,
        "keys_present": False,
        "delegation_relevant": False,
        "instruction_clear": False,
        "overall_score": 0.0
    }
    
    # Format JSON
    if isinstance(response, dict):
        scores["format_valid"] = True
    
    # Clés présentes
    required_keys = ["delegated_agent", "instruction"]
    if all(k in response for k in required_keys):
        scores["keys_present"] = True
    
    # Délégation pertinente
    delegated = response.get("delegated_agent", "").lower()
    if delegated in ["researcher", "code_writer", "critic", "finished"]:
        scores["delegation_relevant"] = True
    
    # Instruction claire
    instruction = response.get("instruction", "")
    if instruction and len(instruction) > 10:
        scores["instruction_clear"] = True
    
    # Score global
    scores["overall_score"] = sum([
        scores["format_valid"],
        scores["keys_present"],
        scores["delegation_relevant"],
        scores["instruction_clear"]
    ]) / 4.0
    
    return scores

def evaluate_researcher(response: dict, query: str) -> dict:
    """
    Évalue la qualité d'une réponse du Researcher.
    
    Critères :
    - Format JSON valide
    - Clés présentes
    - Pertinence de la requête de recherche
    - Présence d'une réponse finale
    """
    scores = {
        "format_valid": False,
        "keys_present": False,
        "query_relevant": False,
        "has_answer": False,
        "overall_score": 0.0
    }
    
    # Format JSON
    if isinstance(response, dict):
        scores["format_valid"] = True
    
    # Clés présentes
    required_keys = ["research_query", "final_answer"]
    if all(k in response for k in required_keys):
        scores["keys_present"] = True
    
    # Requête pertinente
    research_query = response.get("research_query", "").lower()
    query_lower = query.lower()
    # Vérifier si des mots-clés de la requête sont dans la recherche
    query_words = set(query_lower.split())
    research_words = set(research_query.split())
    if query_words.intersection(research_words):
        scores["query_relevant"] = True
    
    # Réponse finale présente
    final_answer = response.get("final_answer")
    if final_answer and str(final_answer).strip() and str(final_answer).lower() != "null":
        scores["has_answer"] = True
    
    # Score global
    scores["overall_score"] = sum([
        scores["format_valid"],
        scores["keys_present"],
        scores["query_relevant"],
        scores["has_answer"]
    ]) / 4.0
    
    return scores

def evaluate_code_writer(response: dict, query: str) -> dict:
    """
    Évalue la qualité d'une réponse du Code Writer.
    
    Critères :
    - Format JSON valide
    - Clés présentes
    - Code Python valide (syntaxe)
    - Code pertinent (correspond à la requête)
    """
    scores = {
        "format_valid": False,
        "keys_present": False,
        "code_valid": False,
        "code_relevant": False,
        "overall_score": 0.0
    }
    
    # Format JSON
    if isinstance(response, dict):
        scores["format_valid"] = True
    
    # Clés présentes
    required_keys = ["python_code", "result_explanation"]
    if all(k in response for k in required_keys):
        scores["keys_present"] = True
    
    # Code Python valide (vérification basique)
    python_code = response.get("python_code", "")
    if python_code:
        # Vérifier la syntaxe de base
        try:
            compile(python_code, '<string>', 'exec')
            scores["code_valid"] = True
        except SyntaxError:
            # Code invalide
            pass
    
    # Code pertinent
    query_lower = query.lower()
    code_lower = python_code.lower()
    
    # Vérifier si des mots-clés de la requête sont dans le code
    if any(keyword in code_lower for keyword in ["discount", "remise", "prix", "price", "calculate", "calcul"]):
        scores["code_relevant"] = True
    
    # Score global
    scores["overall_score"] = sum([
        scores["format_valid"],
        scores["keys_present"],
        scores["code_valid"],
        scores["code_relevant"]
    ]) / 4.0
    
    return scores

def evaluate_critic(response: dict, query: str) -> dict:
    """
    Évalue la qualité d'une réponse du Critic.
    
    Critères :
    - Format JSON valide
    - Clés présentes
    - Critique booléenne valide
    - Suggestions présentes (si critique_ok = False)
    """
    scores = {
        "format_valid": False,
        "keys_present": False,
        "critique_valid": False,
        "suggestions_present": False,
        "overall_score": 0.0
    }
    
    # Format JSON
    if isinstance(response, dict):
        scores["format_valid"] = True
    
    # Clés présentes
    required_keys = ["critique_ok", "suggestions"]
    if all(k in response for k in required_keys):
        scores["keys_present"] = True
    
    # Critique booléenne valide
    critique_ok = response.get("critique_ok")
    if isinstance(critique_ok, bool):
        scores["critique_valid"] = True
    
    # Suggestions présentes (si critique_ok = False)
    suggestions = response.get("suggestions", "")
    if critique_ok is False:
        if suggestions and len(suggestions.strip()) > 0:
            scores["suggestions_present"] = True
    else:
        # Si critique_ok = True, suggestions peut être vide
        scores["suggestions_present"] = True
    
    # Score global
    scores["overall_score"] = sum([
        scores["format_valid"],
        scores["keys_present"],
        scores["critique_valid"],
        scores["suggestions_present"]
    ]) / 4.0
    
    return scores

def evaluate_agent_response(agent_name: str, response: dict, query: str) -> dict:
    """
    Évalue la qualité d'une réponse d'un agent.
    """
    evaluators = {
        "orchestrator": evaluate_orchestrator,
        "researcher": evaluate_researcher,
        "code_writer": evaluate_code_writer,
        "critic": evaluate_critic
    }
    
    evaluator = evaluators.get(agent_name)
    if evaluator:
        return evaluator(response, query)
    else:
        return {"error": f"Évaluateur inconnu pour {agent_name}"}

def test_agent_quality(agent_name: str, query: str, epoch: int = 20):
    """
    Teste un agent et évalue la qualité de sa réponse.
    """
    from test_magrpo_agent import test_magrpo_agent
    
    print(f"\n{'='*70}")
    print(f"📊 Évaluation de la Qualité : {agent_name.upper()}")
    print(f"{'='*70}")
    print(f"Requête: {query}\n")
    
    # Tester l'agent
    result = test_magrpo_agent(agent_name, query, epoch)
    
    if result is None:
        print("❌ Impossible d'évaluer : réponse None")
        return None
    
    # Évaluer la qualité
    scores = evaluate_agent_response(agent_name, result, query)
    
    print(f"\n{'='*70}")
    print("📈 Scores de Qualité")
    print(f"{'='*70}\n")
    
    for criterion, value in scores.items():
        if criterion == "overall_score":
            print(f"{'Score Global':<30} {value:.2%}")
        elif isinstance(value, bool):
            status = "✅" if value else "❌"
            print(f"{criterion.replace('_', ' ').title():<30} {status}")
    
    print(f"\n{'='*70}")
    print(f"📊 Score Global : {scores['overall_score']:.2%}")
    
    if scores['overall_score'] >= 0.75:
        print("✅ Qualité EXCELLENTE")
    elif scores['overall_score'] >= 0.50:
        print("⚠️  Qualité BONNE mais peut être améliorée")
    else:
        print("❌ Qualité INSUFFISANTE - Amélioration nécessaire")
    
    return scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluer la qualité des réponses des agents")
    parser.add_argument("--agent", type=str, help="Nom de l'agent")
    parser.add_argument("--query", type=str, help="Requête de test")
    parser.add_argument("--epoch", type=int, default=20, help="Époque du checkpoint")
    parser.add_argument("--all", action="store_true", help="Tester tous les agents")
    
    args = parser.parse_args()
    
    test_queries = {
        "orchestrator": "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15.",
        "researcher": "Cherche la date de sortie exacte du Google Pixel 8 Pro.",
        "code_writer": "Fais un script Python pour calculer une remise de 15% sur un prix de 899€.",
        "critic": "Évalue ceci : 'Le smartphone est cher mais puissant'."
    }
    
    if args.all:
        all_scores = {}
        for agent_name, query in test_queries.items():
            scores = test_agent_quality(agent_name, query, args.epoch)
            if scores:
                all_scores[agent_name] = scores
            print("\n" + "="*70 + "\n")
        
        # Résumé global
        print("\n" + "="*70)
        print("📊 RÉSUMÉ GLOBAL")
        print("="*70 + "\n")
        
        for agent_name, scores in all_scores.items():
            print(f"{agent_name.upper():<20} Score: {scores['overall_score']:.2%}")
        
        avg_score = sum(s['overall_score'] for s in all_scores.values()) / len(all_scores)
        print(f"\n{'Moyenne':<20} Score: {avg_score:.2%}")
        
    elif args.agent and args.query:
        test_agent_quality(args.agent, args.query, args.epoch)
    else:
        print("Usage:")
        print("  python evaluate_response_quality.py --all")
        print("  python evaluate_response_quality.py --agent orchestrator --query 'Votre requête'")

