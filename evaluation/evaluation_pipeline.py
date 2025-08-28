import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path
import asyncio
import random
from dataclasses import dataclass
from config_ac import settings
from pecc_assistant_hybrid import PECCAssistant
@dataclass
class EvaluationResult:
    hit_rate: float
    mrr: float
    ndcg: float
    precision_at_k: float
    recall_at_k: float
    avg_retrieval_time: float
    config: Dict

class RAGEvaluator:
    """Framework d'évaluation pour le système RAG PECC"""
    
    def __init__(self, config_path: str = "evaluation_config.json"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.evaluation_data = []
        self.ground_truth_pairs = []
        
    def generate_question_context_pairs(self, assistant: PECCAssistant, num_pairs: int = 100) -> List[Dict]:
        """Génère des paires question-contexte pour l'évaluation"""
        
        self.logger.info(f"Génération de {num_pairs} paires question-contexte...")
        
        question_templates = {
            "municipality_specific": [
                "Quelles sont les mesures prioritaires pour {municipality}?",
                "Comment {municipality} peut-elle améliorer son efficacité énergétique?",
                "Quelles subventions sont disponibles pour {municipality}?",
                "Quel est le plan climat de {municipality}?",
                "Comment {municipality} peut-elle réduire ses émissions de transport?"
            ],
            "sector_specific": [
                "Quelles sont les meilleures pratiques pour le secteur {sector}?",
                "Comment optimiser {sector} dans une commune?",
                "Quelles mesures prioritaires pour {sector}?",
                "Comment mesurer l'efficacité des actions {sector}?",
                "Quels sont les coûts typiques pour {sector}?"
            ],
            "canton_specific": [
                "Quel est le plan climat du canton {canton}?",
                "Quelles mesures le canton {canton} recommande-t-il?",
                "Comment les communes {canton} peuvent-elles collaborer?",
                "Quelles sont les spécificités climatiques du {canton}?",
                "Quels financements propose le canton {canton}?"
            ],
            "general": [
                "Comment créer un PECC efficace?",
                "Quelles sont les étapes clés d'un plan climat?",
                "Comment impliquer les citoyens dans les mesures climatiques?",
                "Quels indicateurs utiliser pour suivre les progrès?",
                "Comment prioriser les mesures climatiques?"
            ]
        }
        
        pairs = []
        
        municipalities = self._get_sample_municipalities(assistant)
        sectors = assistant.get_available_sectors()
        cantons = assistant.get_available_cantons()
        
        pairs_per_type = num_pairs // 4
        
        for question_type, templates in question_templates.items():
            for _ in range(pairs_per_type):
                if question_type == "municipality_specific" and municipalities:
                    municipality = random.choice(municipalities)
                    template = random.choice(templates)
                    question = template.format(municipality=municipality)
                    expected_contexts = self._get_expected_contexts_for_municipality(assistant, municipality)
                    
                elif question_type == "sector_specific" and sectors:
                    sector = random.choice(sectors)
                    template = random.choice(templates)
                    question = template.format(sector=sector.lower())
                    expected_contexts = self._get_expected_contexts_for_sector(assistant, sector)
                    
                elif question_type == "canton_specific" and cantons:
                    canton = random.choice(cantons)
                    template = random.choice(templates)
                    question = template.format(canton=canton)
                    expected_contexts = self._get_expected_contexts_for_canton(assistant, canton)
                    
                elif question_type == "general":
                    template = random.choice(templates)
                    question = template
                    expected_contexts = self._get_expected_contexts_general(assistant)
                else:
                    continue
                
                pair = {
                    "question": question,
                    "question_type": question_type,
                    "expected_contexts": expected_contexts,
                    "metadata": {
                        "municipality": municipality if question_type == "municipality_specific" else None,
                        "sector": sector if question_type == "sector_specific" else None,
                        "canton": canton if question_type == "canton_specific" else None
                    }
                }
                pairs.append(pair)
        
        real_questions = self._get_real_user_questions()
        pairs.extend(real_questions[:num_pairs - len(pairs)])
        
        self._save_question_context_pairs(pairs)
        
        self.logger.info(f"Généré {len(pairs)} paires question-contexte")
        return pairs
    
    def _get_sample_municipalities(self, assistant: PECCAssistant) -> List[str]:
        """Récupère un échantillon de communes"""
        if hasattr(assistant.document_processor, 'municipalities_data') and assistant.document_processor.municipalities_data:
            municipalities = list(assistant.document_processor.municipalities_data.keys())
            return random.sample(municipalities, min(20, len(municipalities)))
    
    def _get_expected_contexts_for_municipality(self, assistant: PECCAssistant, municipality: str) -> List[str]:
        """Récupère les contextes attendus pour une commune"""
        try:
            search_queries = [
                f"{municipality}",
                f"commune {municipality}",
                f"{municipality} plan climat",
                f"{municipality} mesures"
            ]
            
            expected_contexts = set()
            for query in search_queries:
                results = assistant.vector_store.search(query, k=5)
                for result in results:
                    if result["similarity_score"] > 0.7:  
                        expected_contexts.add(result["chunk_id"])
            
            return list(expected_contexts)
        except:
            return []
    
    def _get_expected_contexts_for_sector(self, assistant: PECCAssistant, sector: str) -> List[str]:
        """Récupère les contextes attendus pour un secteur"""
        try:
            results = assistant.vector_store.search_by_sector(f"mesures {sector.lower()}", sector, k=10)
            return [r["chunk_id"] for r in results if r["similarity_score"] > 0.6]
        except:
            return []
    
    def _get_expected_contexts_for_canton(self, assistant: PECCAssistant, canton: str) -> List[str]:
        """Récupère les contextes attendus pour un canton"""
        try:
            results = assistant.vector_store.search_by_canton(f"plan climat {canton}", canton, k=10)
            return [r["chunk_id"] for r in results if r["similarity_score"] > 0.6]
        except:
            return []
    
    def _get_expected_contexts_general(self, assistant: PECCAssistant) -> List[str]:
        """Récupère les contextes attendus pour les questions générales"""
        try:
            results = assistant.vector_store.search_by_document_type("mesures prioritaires PECC", "mesure_prioritaire", k=10)
            return [r["chunk_id"] for r in results if r["similarity_score"] > 0.6]
        except:
            return []
    
    def _get_real_user_questions(self) -> List[Dict]:
        """Récupère des questions réelles d'utilisateurs si disponibles"""
        real_questions = [
            {
                "question": "Je fais partie de la commune de Choulex, quelles subventions puis-je obtenir pour la rénovation des bâtiments publics?",
                "question_type": "municipality_specific",
                "expected_contexts": [],  
                "metadata": {"municipality": "Choulex", "sector": "Bâtiment", "canton": None}
            },
            {
                "question": "Comment développer l'alimentation durable dans ma commune?",
                "question_type": "sector_specific", 
                "expected_contexts": [],
                "metadata": {"municipality": None, "sector": "Alimentation", "canton": None}
            },
            {
                "question": "Quelles sont les mesures d'adaptation au changement climatique en Valais?",
                "question_type": "canton_specific",
                "expected_contexts": [],
                "metadata": {"municipality": None, "sector": None, "canton": "Valais"}
            }
        ]
        
        return real_questions
    
    def _save_question_context_pairs(self, pairs: List[Dict]):
        """Sauvegarde les paires question-contexte"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"question_context_pairs_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Paires sauvegardées dans {filename}")
    
    def load_question_context_pairs(self, filename: str) -> List[Dict]:
        """Charge les paires question-contexte depuis un fichier"""
        with open(filename, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        
        self.ground_truth_pairs = pairs
        self.logger.info(f"Chargé {len(pairs)} paires depuis {filename}")
        return pairs
    
    def calculate_hit_rate(self, retrieved_contexts: List[str], expected_contexts: List[str]) -> float:
        """Calcule le hit rate (rappel à k)"""
        if not expected_contexts:
            return 1.0  
        
        hits = len(set(retrieved_contexts) & set(expected_contexts))
        return hits / len(expected_contexts)
    
    def calculate_mrr(self, retrieved_contexts: List[str], expected_contexts: List[str]) -> float:
        """Calcule le Mean Reciprocal Rank"""
        if not expected_contexts:
            return 1.0
        
        for i, context in enumerate(retrieved_contexts):
            if context in expected_contexts:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_ndcg(self, retrieved_contexts: List[str], expected_contexts: List[str], k: int = 5) -> float:
        """Calcule le Normalized Discounted Cumulative Gain"""
        if not expected_contexts:
            return 1.0
        
        dcg = 0.0
        for i, context in enumerate(retrieved_contexts[:k]):
            if context in expected_contexts:
                dcg += 1.0 / np.log2(i + 2)  
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected_contexts), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_precision_at_k(self, retrieved_contexts: List[str], expected_contexts: List[str], k: int = 5) -> float:
        """Calcule la précision à k"""
        if not retrieved_contexts[:k]:
            return 0.0
        
        relevant_retrieved = len(set(retrieved_contexts[:k]) & set(expected_contexts))
        return relevant_retrieved / min(k, len(retrieved_contexts))
    
    async def evaluate_retrieval_config(self, 
                                      assistant: PECCAssistant,
                                      question_pairs: List[Dict],
                                      retrieval_config: Dict) -> EvaluationResult:
        """Évalue une configuration de récupération"""
        
        self.logger.info(f"Évaluation avec config: {retrieval_config}")
        
        hit_rates = []
        mrr_scores = []
        ndcg_scores = []
        precision_scores = []
        retrieval_times = []
        
        for pair in question_pairs:
            question = pair["question"]
            expected_contexts = pair["expected_contexts"]
            
            start_time = datetime.now()
            
            retrieved_docs = await self._retrieve_with_config(assistant, question, retrieval_config)
            retrieved_contexts = [doc["chunk_id"] for doc in retrieved_docs]
            
            end_time = datetime.now()
            retrieval_time = (end_time - start_time).total_seconds()
            
            hit_rate = self.calculate_hit_rate(retrieved_contexts, expected_contexts)
            mrr = self.calculate_mrr(retrieved_contexts, expected_contexts)
            ndcg = self.calculate_ndcg(retrieved_contexts, expected_contexts, retrieval_config.get("k", 5))
            precision = self.calculate_precision_at_k(retrieved_contexts, expected_contexts, retrieval_config.get("k", 5))
            
            hit_rates.append(hit_rate)
            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            precision_scores.append(precision)
            retrieval_times.append(retrieval_time)
        
        result = EvaluationResult(
            hit_rate=np.mean(hit_rates),
            mrr=np.mean(mrr_scores),
            ndcg=np.mean(ndcg_scores),
            precision_at_k=np.mean(precision_scores),
            recall_at_k=np.mean(hit_rates),  # Hit rate = Recall
            avg_retrieval_time=np.mean(retrieval_times),
            config=retrieval_config
        )
        
        return result
    
    async def _retrieve_with_config(self, assistant: PECCAssistant, query: str, config: Dict) -> List[Dict]:
        """Récupère des documents avec une configuration spécifique"""
        k = config.get("k", 5)
        search_method = config.get("search_method", "standard")
        
        analysis = assistant._analyze_query(query)
        
        if search_method == "standard":
            if analysis["type"] == "municipality_specific" and analysis.get("canton"):
                return assistant.vector_store.search_by_canton(query, analysis["canton"], k=k)
            elif analysis["type"] == "sector_specific" and analysis.get("sector"):
                return assistant.vector_store.search_by_sector(query, analysis["sector"], k=k)
            else:
                return assistant.vector_store.search(query, k=k)
        
        return assistant.vector_store.search(query, k=k)
    
    def compare_retrieval_configs(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Compare les résultats de différentes configurations"""
        
        comparison_data = []
        for result in results:
            comparison_data.append({
                "Configuration": str(result.config),
                "Hit Rate": f"{result.hit_rate:.3f}",
                "MRR": f"{result.mrr:.3f}",
                "NDCG": f"{result.ndcg:.3f}",
                "Precision@K": f"{result.precision_at_k:.3f}",
                "Recall@K": f"{result.recall_at_k:.3f}",
                "Avg Retrieval Time (s)": f"{result.avg_retrieval_time:.3f}",
                "Overall Score": f"{(result.hit_rate + result.mrr + result.ndcg) / 3:.3f}"
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values("Overall Score", ascending=False)
    
    def save_evaluation_results(self, results: List[EvaluationResult], filename: str = None):
        """Sauvegarde les résultats d'évaluation"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        # Convertit les résultats en format sérialisable
        serializable_results = []
        for result in results:
            serializable_results.append({
                "hit_rate": result.hit_rate,
                "mrr": result.mrr,
                "ndcg": result.ndcg,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "avg_retrieval_time": result.avg_retrieval_time,
                "config": result.config,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Résultats sauvegardés dans {filename}")

# Exemple d'utilisation
async def run_evaluation_example():
    """Exemple d'utilisation du framework d'évaluation"""
    
    # Initialise l'assistant
    assistant = PECCAssistant(settings)
    await assistant.initialize()
    
    # Crée l'évaluateur
    evaluator = RAGEvaluator()
    
    # Génère les paires question-contexte
    question_pairs = evaluator.generate_question_context_pairs(assistant, num_pairs=50)
    
    # Définit différentes configurations à tester
    configs_to_test = [
        {"k": 3, "search_method": "standard"},
        {"k": 5, "search_method": "standard"},
        {"k": 7, "search_method": "standard"},
        {"k": 10, "search_method": "standard"}
    ]
    
    # Évalue chaque configuration
    results = []
    for config in configs_to_test:
        result = await evaluator.evaluate_retrieval_config(assistant, question_pairs, config)
        results.append(result)
        print(f"Config {config}: Hit Rate={result.hit_rate:.3f}, MRR={result.mrr:.3f}")
    
    # Compare les résultats
    comparison_df = evaluator.compare_retrieval_configs(results)
    print("\nComparaison des configurations:")
    print(comparison_df)
    
    # Sauvegarde les résultats
    evaluator.save_evaluation_results(results)

if __name__ == "__main__":
    asyncio.run(run_evaluation_example())