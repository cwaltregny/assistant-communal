import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import argparse

sys.path.append(str(Path(__file__).parent))

from config_ac import settings
from pecc_assistant_hybrid import PECCAssistant as advanced_assistant
from pecc_assistant_chat_v2 import PECCAssistant as simple_assistant
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pecc_rag_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PECCRAGEvaluator:
    """
    Evaluator of the PECC Assistant RAG systems
    """
    
    def __init__(self, dataset1_path: str, dataset2_path: str):
        self.dataset1 = self.load_openai_dataset(dataset1_path)
        self.dataset2 = self.load_openai_dataset(dataset2_path)
        self.logger = logging.getLogger(__name__)
        
    def load_openai_dataset(self, path: str) -> List[Dict]:
        """Load OpenAI-generated dataset from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'questions' in data:
            questions = data['questions']
        else:
            questions = data  
        
        logger.info(f"Loaded {len(questions)} questions from {path}")
        if 'metadata' in data:
            logger.info(f"Dataset metadata: {data['metadata']}")
        
        return questions
    
    def calculate_mrr(self, retrieved_docs: List[List[str]], 
                     ground_truth: List[Set[str]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for query_results, relevant_docs in zip(retrieved_docs, ground_truth):
            rank = 0
            for i, doc_id in enumerate(query_results):
                if doc_id in relevant_docs:
                    rank = 1 / (i + 1)
                    break
            reciprocal_ranks.append(rank)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
    def calculate_hit_rate(self, retrieved_docs: List[List[str]], 
                          ground_truth: List[Set[str]], k: int = 10) -> float:
        """Calculate Hit Rate @ k"""
        hits = 0
        
        for query_results, relevant_docs in zip(retrieved_docs, ground_truth):
            top_k_results = set(query_results[:k])
            if top_k_results & relevant_docs:
                hits += 1
        
        return hits / len(retrieved_docs) if retrieved_docs else 0
    
    def prepare_ground_truth_dataset(self, dataset: List[Dict]) -> Tuple[List[str], List[Set[str]]]:
        """Prepare queries and ground truth from OpenAI-generated dataset"""
        queries = []
        ground_truth = []
        
        for item in dataset:
            queries.append(item['question'])
            ground_truth.append(set(item['expected_docs']))
        
        return queries, ground_truth
    
    def extract_doc_ids_from_references(self, references: List[Dict]) -> List[str]:
        """
        Extract document IDs from the references returned by the PECC assistants
        Updated to handle various reference formats
        """
        doc_ids = []
        
        for ref in references:
            doc_id = None
            
            if 'chunk_id' in ref:
                doc_id = ref['chunk_id']
            
            elif 'source_file' in ref:
                source = ref['source_file']
                page = ref.get('page_number', 0)
                
                if source.endswith('.pdf'):
                    base_name = source[:-4]  # Remove .pdf
                else:
                    base_name = source
                
                chunk_num = ref.get('chunk_number', 0)
                doc_id = f"{base_name}_p{page}_c{chunk_num}"
                
                doc_id = doc_id.replace('.pdf', '').replace(' ', '_').lower()
            
            elif 'extract' in ref or 'content' in ref:
                content = ref.get('extract', ref.get('content', ''))
                source = ref.get('source_file', 'unknown')
                doc_id = f"{source}_content_{hash(content) % 10000}"
            
            if doc_id:
                doc_ids.append(doc_id)
        
        return doc_ids
    
    async def evaluate_assistant(self, assistant, dataset_num: int = 1, 
                                k_values: List[int] = [1, 3, 5, 10],
                                max_queries: int = None) -> Dict:
        """
        Evaluate a PECC assistant on specified dataset
        """
        if dataset_num == 1:
            dataset = self.dataset1
        else:
            dataset = self.dataset2
        
        queries, ground_truth = self.prepare_ground_truth_dataset(dataset)
        
        if max_queries:
            queries = queries[:max_queries]
            ground_truth = ground_truth[:max_queries]
            dataset = dataset[:max_queries]
        
        retrieved_docs = []
        processing_errors = 0
        successful_queries = 0
        detailed_results = []
        
        self.logger.info(f"Evaluating {len(queries)} queries on dataset {dataset_num}")
        
        for i, (query, gt, item) in enumerate(zip(queries, ground_truth, dataset)):
            try:
                assistant.reset_conversation()
                
                result = await assistant.process_message(query)
                
                if result.get('status') == 'success':
                    references = result.get('references', [])
                    doc_ids = self.extract_doc_ids_from_references(references)
                    retrieved_docs.append(doc_ids)
                    successful_queries += 1
                    
                    detailed_results.append({
                        'query': query,
                        'expected': list(gt),
                        'retrieved': doc_ids,
                        'matched': list(set(doc_ids) & gt),
                        'category': item.get('category', 'unknown'),
                        'difficulty': item.get('difficulty', 'unknown'),
                        'municipality': item.get('municipality'),
                        'canton': item.get('canton')
                    })
                    
                    if i < 3:
                        self.logger.info(f"Query {i+1}: {query[:50]}...")
                        self.logger.info(f"Retrieved: {doc_ids[:3] if doc_ids else 'None'}...")
                        self.logger.info(f"Expected: {list(gt)[:3]}...")
                        self.logger.info(f"Match: {bool(set(doc_ids) & gt)}")
                else:
                    self.logger.warning(f"Failed to process query {i+1}: {query[:50]}...")
                    retrieved_docs.append([])
                    processing_errors += 1
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(queries)} queries")
                    
            except Exception as e:
                self.logger.error(f"Error processing query {i+1}: {query[:50]}... Error: {e}")
                retrieved_docs.append([])
                processing_errors += 1
        
        mrr = self.calculate_mrr(retrieved_docs, ground_truth)
        
        metrics = {
            'MRR': mrr,
            'processing_errors': processing_errors,
            'successful_queries': successful_queries,
            'total_queries': len(queries),
            'success_rate': successful_queries / len(queries) if queries else 0
        }
        
        for k in k_values:
            hit_rate = self.calculate_hit_rate(retrieved_docs, ground_truth, k)
            metrics[f'Hit_Rate@{k}'] = hit_rate
        
        quality_metrics = self._analyze_retrieval_quality(retrieved_docs, ground_truth)
        metrics.update(quality_metrics)
        
        category_metrics = self._analyze_by_categories(detailed_results)
        metrics['category_analysis'] = category_metrics
        
        return metrics
    
    def _analyze_retrieval_quality(self, retrieved_docs: List[List[str]], 
                                 ground_truth: List[Set[str]]) -> Dict:
        """Analyze the quality of retrieved documents"""
        total_relevant = sum(len(gt) for gt in ground_truth)
        total_retrieved = sum(len(docs) for docs in retrieved_docs)
        
        relevant_retrieved = 0
        perfect_matches = 0
        partial_matches = 0
        no_matches = 0
        
        for docs, gt in zip(retrieved_docs, ground_truth):
            overlap = set(docs) & gt
            relevant_retrieved += len(overlap)
            
            if len(overlap) == len(gt) and len(gt) > 0:
                perfect_matches += 1
            elif len(overlap) > 0:
                partial_matches += 1
            else:
                no_matches += 1
        
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_docs_per_query = total_retrieved / len(retrieved_docs) if retrieved_docs else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'relevant_retrieved': relevant_retrieved,
            'total_relevant': total_relevant,
            'total_retrieved': total_retrieved,
            'avg_docs_per_query': avg_docs_per_query,
            'perfect_matches': perfect_matches,
            'partial_matches': partial_matches,
            'no_matches': no_matches
        }
    
    def _analyze_by_categories(self, detailed_results: List[Dict]) -> Dict:
        """Analyze performance by categories"""
        categories = {}
        
        for result in detailed_results:
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'hits': 0}
            
            categories[category]['total'] += 1
            if result['matched']:  
                categories[category]['hits'] += 1
        
        for category in categories:
            total = categories[category]['total']
            hits = categories[category]['hits']
            categories[category]['hit_rate'] = hits / total if total > 0 else 0
        
        return categories
    
    async def compare_systems(self, assistant1, assistant2, dataset_num: int = 1,
                            k_values: List[int] = [1, 3, 5, 10], 
                            system1_name: str = "advanced_assistant", 
                            system2_name: str = "simple_assistant",
                            max_queries: int = None) -> Tuple[Dict, Dict]:
        """
        Compare two PECC assistant systems
        """
        dataset_name = f"Dataset {dataset_num} ({'OpenAI Generated' if dataset_num == 1 else 'OpenAI Generated #2'})"
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        print(f"\n🔄 Evaluating {system1_name}...")
        metrics1 = await self.evaluate_assistant(assistant1, dataset_num, k_values, max_queries)
        
        print(f"\n🔄 Evaluating {system2_name}...")
        metrics2 = await self.evaluate_assistant(assistant2, dataset_num, k_values, max_queries)
        
        print(f"\n📊 {system1_name} Results:")
        self._print_metrics(metrics1)
        
        print(f"\n📊 {system2_name} Results:")
        self._print_metrics(metrics2)
        
        print(f"\n🔀 Comparison ({system1_name} - {system2_name}):")
        comparison_metrics = ['MRR'] + [f'Hit_Rate@{k}' for k in k_values] + ['precision', 'recall', 'f1_score']
        for metric in comparison_metrics:
            if metric in metrics1 and metric in metrics2:
                diff = metrics1[metric] - metrics2[metric]
                symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                print(f"  {symbol} {metric}: {diff:+.4f}")
        
        # Category comparison
        print(f"\n📋 Category Performance Comparison:")
        cat1 = metrics1.get('category_analysis', {})
        cat2 = metrics2.get('category_analysis', {})
        
        all_categories = set(cat1.keys()) | set(cat2.keys())
        for category in sorted(all_categories):
            rate1 = cat1.get(category, {}).get('hit_rate', 0)
            rate2 = cat2.get(category, {}).get('hit_rate', 0)
            diff = rate1 - rate2
            symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            print(f"  {symbol} {category}: {rate1:.3f} vs {rate2:.3f} (diff: {diff:+.3f})")
        
        return metrics1, metrics2
    
    def _print_metrics(self, metrics: Dict):
        """Pretty print metrics"""
        important_metrics = ['MRR', 'Hit_Rate@1', 'Hit_Rate@5', 'Hit_Rate@10', 'precision', 'recall', 'f1_score']
        
        for metric in important_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    print(f"  📈 {metric}: {value:.4f}")
        
        if 'perfect_matches' in metrics:
            print(f"  ✅ Perfect matches: {metrics['perfect_matches']}")
            print(f"  🔶 Partial matches: {metrics['partial_matches']}")
            print(f"  ❌ No matches: {metrics['no_matches']}")
        
        other_metrics = ['total_queries', 'successful_queries', 'processing_errors', 'success_rate']
        for metric in other_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    print(f"  ℹ️  {metric}: {value:.4f}")
                else:
                    print(f"  ℹ️  {metric}: {value}")
    
    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Results saved to {filename}")

async def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate PECC RAG systems")
    parser.add_argument("--dataset1", default="ground_truth_dataset.json", help="Path to dataset 1 (OpenAI generated)")
    parser.add_argument("--dataset2", default="ground_truth_dataset_fiches.json", help="Path to dataset 2 (OpenAI generated)")
    parser.add_argument("--max-queries", type=int, help="Limit number of queries for testing")
    parser.add_argument("--quick-test", action="store_true", help="Run with only 5 queries per dataset")
    parser.add_argument("--dataset-only", type=int, choices=[1, 2], help="Run only on specified dataset")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.max_queries = 5
    
    logger.info("🚀 Starting PECC RAG evaluation with OpenAI-generated datasets...")
    
    # Initialize evaluator
    try:
        evaluator = PECCRAGEvaluator(args.dataset1, args.dataset2)
        logger.info(f"📊 Loaded datasets: {len(evaluator.dataset1)} and {len(evaluator.dataset2)} questions")
    except Exception as e:
        logger.error(f"❌ Failed to load datasets: {e}")
        return
    
    # Initialize assistants
    logger.info("🔧 Initializing assistants...")
    
    try:
        logger.info("   Initializing Advanced PECCAssistant...")
        simplified_assistant = advanced_assistant(settings)
        await simplified_assistant.initialize()
        logger.info("   ✅ Advanced PECCAssistant ready")
        
        logger.info("   Initializing Simple PECCAssistant...")
        standard_assistant = simple_assistant(settings)
        await standard_assistant.initialize()
        logger.info("   ✅ PECCAssistant ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize assistants: {e}")
        logger.error("   Make sure you have run the embedding generation scripts")
        return
    
    # Run evaluations
    all_results = {}
    
    # Dataset 1 evaluation
    if not args.dataset_only or args.dataset_only == 1:
        logger.info("\n📋 Starting Dataset 1 evaluation...")
        try:
            metrics1_ds1, metrics2_ds1 = await evaluator.compare_systems(
                simplified_assistant, standard_assistant, 
                dataset_num=1,
                system1_name="Advanced PECCAssistant",
                system2_name="Simple PECCAssistant",
                max_queries=args.max_queries
            )
            
            all_results['dataset1_comparison'] = {
                'simplified_assistant': metrics1_ds1,
                'standard_assistant': metrics2_ds1
            }
            
        except Exception as e:
            logger.error(f"❌ Dataset 1 evaluation failed: {e}")
            return
    
    # Dataset 2 evaluation
    if not args.dataset_only or args.dataset_only == 2:
        logger.info("\n📋 Starting Dataset 2 evaluation...")
        try:
            metrics1_ds2, metrics2_ds2 = await evaluator.compare_systems(
                simplified_assistant, standard_assistant, 
                dataset_num=2,
                system1_name="Advanced PECCAssistant",
                system2_name="Simple PECCAssistant",
                max_queries=args.max_queries
            )
            
            all_results['dataset2_comparison'] = {
                'simplified_assistant': metrics1_ds2,
                'standard_assistant': metrics2_ds2
            }
            
        except Exception as e:
            logger.error(f"❌ Dataset 2 evaluation failed: {e}")
    
    print(f"\n{'='*60}")
    print("📋 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    if 'dataset1_comparison' in all_results:
        ds1_simple = all_results['dataset1_comparison']['simplified_assistant']
        ds1_standard = all_results['dataset1_comparison']['standard_assistant']
        
        print(f"\n📊 Dataset 1 (OpenAI Generated) - {len(evaluator.dataset1)} questions:")
        print(f"   AdvancedPECCAssistant: MRR={ds1_simple.get('MRR', 0):.4f}, Hit@5={ds1_simple.get('Hit_Rate@5', 0):.4f}, F1={ds1_simple.get('f1_score', 0):.4f}")
        print(f"   Simple PECCAssistant:          MRR={ds1_standard.get('MRR', 0):.4f}, Hit@5={ds1_standard.get('Hit_Rate@5', 0):.4f}, F1={ds1_standard.get('f1_score', 0):.4f}")
        
        mrr_diff = ds1_simple.get('MRR', 0) - ds1_standard.get('MRR', 0)
        hit5_diff = ds1_simple.get('Hit_Rate@5', 0) - ds1_standard.get('Hit_Rate@5', 0)
        f1_diff = ds1_simple.get('f1_score', 0) - ds1_standard.get('f1_score', 0)
        print(f"   → Difference:           MRR={mrr_diff:+.4f}, Hit@5={hit5_diff:+.4f}, F1={f1_diff:+.4f}")
    
    if 'dataset2_comparison' in all_results:
        ds2_simple = all_results['dataset2_comparison']['simplified_assistant']
        ds2_standard = all_results['dataset2_comparison']['standard_assistant']
        
        print(f"\n📊 Dataset 2 (OpenAI Generated) - {len(evaluator.dataset2)} questions:")
        print(f"   Advanced PECCAssistant: MRR={ds2_simple.get('MRR', 0):.4f}, Hit@5={ds2_simple.get('Hit_Rate@5', 0):.4f}, F1={ds2_simple.get('f1_score', 0):.4f}")
        print(f"   Simple PECCAssistant:          MRR={ds2_standard.get('MRR', 0):.4f}, Hit@5={ds2_standard.get('Hit_Rate@5', 0):.4f}, F1={ds2_standard.get('f1_score', 0):.4f}")
        
        mrr_diff = ds2_simple.get('MRR', 0) - ds2_standard.get('MRR', 0)
        hit5_diff = ds2_simple.get('Hit_Rate@5', 0) - ds2_standard.get('Hit_Rate@5', 0)
        f1_diff = ds2_simple.get('f1_score', 0) - ds2_standard.get('f1_score', 0)
        print(f"   → Difference:           MRR={mrr_diff:+.4f}, Hit@5={hit5_diff:+.4f}, F1={f1_diff:+.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pecc_rag_evaluation_results_{timestamp}.json"
    
    all_results['evaluation_metadata'] = {
        'timestamp': timestamp,
        'datasets': [args.dataset1, args.dataset2],
        'max_queries_per_dataset': args.max_queries,
        'total_questions_evaluated': {
            'dataset1': len(evaluator.dataset1) if args.max_queries is None else min(args.max_queries, len(evaluator.dataset1)),
            'dataset2': len(evaluator.dataset2) if args.max_queries is None else min(args.max_queries, len(evaluator.dataset2))
        },
        'dataset_sources': 'OpenAI GPT-4 generated'
    }
    
    evaluator.save_results(all_results, filename)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📄 Detailed results saved to: {filename}")
    print(f"📄 Logs saved to: pecc_rag_evaluation.log")
    
    if all_results:
        print(f"\n💡 Quick Analysis:")
        if 'dataset1_comparison' in all_results:
            ds1_simple = all_results['dataset1_comparison']['simplified_assistant']
            ds1_standard = all_results['dataset1_comparison']['standard_assistant']
            
            if ds1_simple.get('MRR', 0) > ds1_standard.get('MRR', 0):
                print("   📈 Advanced PECCAssistant shows better MRR on Dataset 1")
            else:
                print("   📈  Simple PECCAssistant shows better MRR on Dataset 1")

if __name__ == "__main__":
    asyncio.run(main())