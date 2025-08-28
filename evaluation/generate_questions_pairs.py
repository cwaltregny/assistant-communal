import json
import asyncio
import logging
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import argparse
from datetime import datetime
import openai
from openai import AsyncOpenAI
import tiktoken
import re
import time
from dataclasses import dataclass
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    chunk_id: str
    content: str
    source_file: str
    page_number: int
    document_type: str
    metadata: Dict
    
@dataclass
class GeneratedQA:
    """Represents a generated question-answer pair"""
    question: str
    answer: str
    relevant_chunks: List[str]
    difficulty: str
    category: str
    context: str
    municipality: Optional[str] = None
    canton: Optional[str] = None

class OpenAIGroundTruthGenerator:
    """
    Generates ground truth question-answer pairs using OpenAI
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # PECC-specific categories and municipalities
        self.sectors = [
            "Transport", "Bâtiment", "Energie", "Alimentation", "Aménagement du territoire"
        ]
        
        self.cantons = ["Genève", "Vaud", "Neuchâtel", "Fribourg", "Valais"]
        
        self.municipalities = [
            "Choulex", "Satigny", "Puplinge", "Onex",  # Genève
            "Corseaux", "Treytorrens", "Noville", "Lonay", "Begnins", "Renens",  # Vaud
            "Laténa",  # Neuchâtel
            "Gruyères", "Courtepin", "Billens-Hennens", "Estavayer",  # Fribourg
            "Leytron", "Salvan", "Grächen"  # Valais
        ]
        
        self.difficulty_levels = ["easy", "medium", "hard"]
        
    async def load_document_chunks(self, chunks_file: str) -> List[DocumentChunk]:
        """Load document chunks from processed documents"""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = []
            for chunk_data in chunks_data:
                chunk = DocumentChunk(
                    chunk_id=chunk_data.get('chunk_id', f"chunk_{len(chunks)}"),
                    content=chunk_data.get('content', ''),
                    source_file=chunk_data.get('source_file', ''),
                    page_number=chunk_data.get('page_number', 0),
                    document_type=chunk_data.get('document_type', 'unknown'),
                    metadata=chunk_data.get('metadata', {})
                )
                chunks.append(chunk)
            
            logger.info(f"Loaded {len(chunks)} document chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_content(self, content: str, max_tokens: int = 3000) -> str:
        """Truncate content to fit within token limit"""
        tokens = self.encoding.encode(content)
        if len(tokens) <= max_tokens:
            return content
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    async def generate_questions_from_chunk(self, chunk: DocumentChunk, 
                                          num_questions: int = 3) -> List[GeneratedQA]:
        """Generate questions from a single document chunk"""
        
        # Determine context and specificity
        context_info = self._analyze_chunk_context(chunk)
        
        prompt = f"""
Tu es un expert en politique climatique suisse et en Plans Energie Climat Communaux (PECC).

Génère {num_questions} questions de qualité différente basées sur le document suivant:

**Source:** {chunk.source_file}
**Type:** {chunk.document_type}
**Contenu:**
{self.truncate_content(chunk.content, 2500)}

**Contexte détecté:**
- Canton: {context_info.get('canton', 'Général')}
- Commune: {context_info.get('municipality', 'Général')}
- Secteur: {context_info.get('sector', 'Général')}

**Instructions:**
1. Génère des questions de 3 niveaux de difficulté (easy, medium, hard)
2. Les questions doivent être spécifiques au contenu du document
3. Varie les types de questions: factuelles, analytiques, pratiques
4. Pour les questions spécifiques à une commune, utilise le contexte approprié
5. Assure-toi que les réponses sont trouvables dans le document

**Format de réponse (JSON):**
[
  {{
    "question": "Question en français",
    "answer": "Réponse détaillée basée sur le document",
    "difficulty": "easy|medium|hard",
    "category": "transport|batiment|energie|alimentation|amenagement",
    "context": "en Suisse|dans le canton de X|à Y",
    "municipality": null ou "nom de la commune",
    "canton": null ou "nom du canton"
  }}
]

Génère maintenant les questions:
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en PECC et génération de questions d'évaluation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group())
                
                generated_qas = []
                for q_data in questions_data:
                    qa = GeneratedQA(
                        question=q_data['question'],
                        answer=q_data['answer'],
                        relevant_chunks=[chunk.chunk_id],
                        difficulty=q_data.get('difficulty', 'medium'),
                        category=q_data.get('category', 'general'),
                        context=q_data.get('context', 'en Suisse'),
                        municipality=q_data.get('municipality'),
                        canton=q_data.get('canton')
                    )
                    generated_qas.append(qa)
                
                return generated_qas
            else:
                logger.warning(f"No JSON found in response for chunk {chunk.chunk_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating questions for chunk {chunk.chunk_id}: {e}")
            return []
    
    def _analyze_chunk_context(self, chunk: DocumentChunk) -> Dict:
        """Analyze chunk to determine context (canton, municipality, sector)"""
        content_lower = chunk.content.lower()
        source_lower = chunk.source_file.lower()
        
        context = {}
        
        # Detect canton
        for canton in self.cantons:
            if canton.lower() in content_lower or canton.lower() in source_lower:
                context['canton'] = canton
                break
        
        # Detect municipality
        for municipality in self.municipalities:
            if municipality.lower() in content_lower:
                context['municipality'] = municipality
                break
        
        # Detect sector
        for sector in self.sectors:
            if sector.lower() in content_lower or sector.lower() in source_lower:
                context['sector'] = sector
                break
        
        return context
    
    async def identify_relevant_documents(self, question: str, answer: str, 
                                        all_chunks: List[DocumentChunk]) -> List[str]:
        """
        Use OpenAI to identify which documents are relevant to a question
        """
        # Create a summary of available documents
        doc_summaries = []
        for chunk in all_chunks[:50]:  # Limit for token constraints
            summary = f"{chunk.chunk_id}: {chunk.source_file} (p{chunk.page_number}) - {chunk.content[:150]}..."
            doc_summaries.append(summary)
        
        docs_text = "\n".join(doc_summaries)
        
        prompt = f"""
Tu dois identifier quels documents sont pertinents pour répondre à cette question.

**Question:** {question}
**Réponse attendue:** {answer}

**Documents disponibles:**
{self.truncate_content(docs_text, 3000)}

**Instructions:**
1. Identifie les 3-5 documents les plus pertinents pour cette question
2. Un document est pertinent s'il contient des informations nécessaires pour répondre
3. Priorise les documents qui contiennent des informations directement liées
4. Retourne uniquement les chunk_id des documents pertinents

**Format de réponse (JSON):**
["chunk_id_1", "chunk_id_2", "chunk_id_3"]

Chunk IDs pertinents:
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en analyse de pertinence documentaire."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON array
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                chunk_ids = json.loads(json_match.group())
                return [cid for cid in chunk_ids if isinstance(cid, str)]
            else:
                logger.warning(f"No JSON found in relevance response for question: {question[:50]}...")
                return []
                
        except Exception as e:
            logger.error(f"Error identifying relevant documents: {e}")
            return []
    
    async def generate_cross_document_questions(self, chunks: List[DocumentChunk], 
                                              num_questions: int = 10) -> List[GeneratedQA]:
        """Generate questions that require multiple documents to answer"""
        
        # Group chunks by type and canton for cross-document questions
        canton_groups = {}
        sector_groups = {}
        
        for chunk in chunks:
            context = self._analyze_chunk_context(chunk)
            
            if context.get('canton'):
                canton = context['canton']
                if canton not in canton_groups:
                    canton_groups[canton] = []
                canton_groups[canton].append(chunk)
            
            if context.get('sector'):
                sector = context['sector']
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(chunk)
        
        cross_doc_questions = []
        
        # Generate comparative questions across cantons
        for canton, canton_chunks in canton_groups.items():
            if len(canton_chunks) >= 2:
                sample_chunks = random.sample(canton_chunks, min(3, len(canton_chunks)))
                questions = await self._generate_comparative_questions(
                    sample_chunks, f"canton de {canton}", "canton_comparison"
                )
                cross_doc_questions.extend(questions)
        
        # Generate sector-specific questions across documents
        for sector, sector_chunks in sector_groups.items():
            if len(sector_chunks) >= 2:
                sample_chunks = random.sample(sector_chunks, min(3, len(sector_chunks)))
                questions = await self._generate_comparative_questions(
                    sample_chunks, f"secteur {sector}", "sector_analysis"
                )
                cross_doc_questions.extend(questions[:2])  # Limit per sector
        
        return cross_doc_questions[:num_questions]
    
    async def _generate_comparative_questions(self, chunks: List[DocumentChunk], 
                                            context_desc: str, 
                                            question_type: str) -> List[GeneratedQA]:
        """Generate questions that compare or synthesize across multiple chunks"""
        
        chunks_text = "\n\n".join([
            f"**Document {i+1} ({chunk.source_file}):**\n{chunk.content[:500]}..."
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""
Génère 2 questions qui nécessitent de combiner les informations de plusieurs documents pour le {context_desc}.

**Documents source:**
{self.truncate_content(chunks_text, 2500)}

**Type de questions à générer:** {question_type}

**Instructions:**
1. Les questions doivent nécessiter une synthèse des informations de plusieurs documents
2. Une question de difficulté "medium" et une "hard"
3. Les questions doivent être pratiques et utiles pour les communes
4. Assure-toi que les réponses sont trouvables en combinant les documents

**Format de réponse (JSON):**
[
  {{
    "question": "Question comparative/synthétique",
    "answer": "Réponse basée sur la synthèse des documents",
    "difficulty": "medium|hard",
    "category": "amenagement|transport|energie|batiment|alimentation",
    "context": "contexte approprié",
    "municipality": null,
    "canton": "canton si applicable"
  }}
]

Questions:
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en synthèse documentaire pour les PECC."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                questions_data = json.loads(json_match.group())
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                
                generated_qas = []
                for q_data in questions_data:
                    qa = GeneratedQA(
                        question=q_data['question'],
                        answer=q_data['answer'],
                        relevant_chunks=chunk_ids,
                        difficulty=q_data.get('difficulty', 'medium'),
                        category=q_data.get('category', 'amenagement'),
                        context=q_data.get('context', f"dans le {context_desc}"),
                        municipality=q_data.get('municipality'),
                        canton=q_data.get('canton')
                    )
                    generated_qas.append(qa)
                
                return generated_qas
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating comparative questions: {e}")
            return []
    
    async def enhance_existing_dataset(self, existing_file: str, 
                                     all_chunks: List[DocumentChunk]) -> List[Dict]:
        """Enhance existing dataset by finding better relevant documents"""
        
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        enhanced_data = []
        
        for i, item in enumerate(existing_data):
            logger.info(f"Enhancing item {i+1}/{len(existing_data)}")
            
            question = item['question']
            
            # Generate answer if not present
            if 'answer' not in item:
                answer = await self._generate_answer_for_question(question, all_chunks)
                item['answer'] = answer
            
            # Find better relevant documents
            relevant_docs = await self.identify_relevant_documents(
                question, item.get('answer', ''), all_chunks
            )
            
            # Update the item
            enhanced_item = item.copy()
            enhanced_item['expected_docs'] = relevant_docs
            if 'answer' not in enhanced_item:
                enhanced_item['answer'] = item.get('answer', '')
            
            enhanced_data.append(enhanced_item)
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        return enhanced_data
    
    async def _generate_answer_for_question(self, question: str, 
                                          chunks: List[DocumentChunk]) -> str:
        """Generate an answer for a question using the document chunks"""
        
        # Find most relevant chunks for the question
        relevant_content = []
        for chunk in chunks[:20]:  # Limit for token constraints
            if any(word in chunk.content.lower() for word in question.lower().split() if len(word) > 3):
                relevant_content.append(f"{chunk.source_file}: {chunk.content[:300]}...")
        
        context_text = "\n\n".join(relevant_content[:5])
        
        prompt = f"""
Réponds à cette question en te basant sur les documents fournis.

**Question:** {question}

**Documents disponibles:**
{self.truncate_content(context_text, 2000)}

**Instructions:**
- Réponds de manière précise et factuelle
- Base ta réponse uniquement sur les informations des documents
- Si les documents ne contiennent pas assez d'informations, dis-le
- Donne une réponse structurée et pratique

**Réponse:**
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en PECC qui répond précisément aux questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Réponse non générée en raison d'une erreur."
    
    def format_for_evaluation(self, generated_qas: List[GeneratedQA]) -> List[Dict]:
        """Format generated QAs for evaluation dataset"""
        
        evaluation_data = []
        
        for qa in generated_qas:
            item = {
                "question": qa.question,
                "expected_docs": qa.relevant_chunks,
                "context": qa.context,
                "difficulty": qa.difficulty,
                "category": qa.category,
                "municipality": qa.municipality,
                "canton": qa.canton,
                "answer": qa.answer  # Include generated answer
            }
            evaluation_data.append(item)
        
        return evaluation_data
    
    async def generate_complete_dataset(self, chunks_file: str, 
                                      output_file: str,
                                      questions_per_chunk: int = 2,
                                      cross_doc_questions: int = 15,
                                      enhance_existing: str = None) -> Dict:
        """Generate a complete evaluation dataset"""
        
        logger.info("🚀 Starting ground truth generation...")
        
        # Load document chunks
        chunks = await self.load_document_chunks(chunks_file)
        if not chunks:
            raise ValueError("No chunks loaded")
        
        all_generated_qas = []
        
        # Generate questions from individual chunks
        if enhance_existing:
            logger.info(f"📋 Enhancing existing dataset: {enhance_existing}")
            enhanced_data = await self.enhance_existing_dataset(enhance_existing, chunks)
            
            # Convert to GeneratedQA format
            for item in enhanced_data:
                qa = GeneratedQA(
                    question=item['question'],
                    answer=item.get('answer', ''),
                    relevant_chunks=item.get('expected_docs', []),
                    difficulty=item.get('difficulty', 'medium'),
                    category=item.get('category', 'general'),
                    context=item.get('context', 'en Suisse'),
                    municipality=item.get('municipality'),
                    canton=item.get('canton')
                )
                all_generated_qas.append(qa)
        
        else:
            logger.info(f"📄 Generating questions from {len(chunks)} chunks...")
            sampled_chunks = []
            for c in chunks:
                if any(w in c.source_file for w in['geneve','valais','vaud','fribourg','neuchatel','jura']):
                    sampled_chunks.append(c)
           
            sampled_chunks = random.sample(sampled_chunks, min(20, len(chunks)))
            
            for i, chunk in enumerate(sampled_chunks):
                logger.info(f"Processing chunk {i+1}/{len(sampled_chunks)}: {chunk.source_file}")
                
                chunk_qas = await self.generate_questions_from_chunk(chunk, questions_per_chunk)
                all_generated_qas.extend(chunk_qas)
                
                # Rate limiting
                await asyncio.sleep(1)
                
                if i % 5 == 4:  # Every 5 chunks
                    logger.info(f"Generated {len(all_generated_qas)} questions so far...")
            
            # Generate cross-document questions
            # logger.info(f"🔗 Generating {cross_doc_questions} cross-document questions...")
            # cross_qas = await self.generate_cross_document_questions(chunks, cross_doc_questions)
            # all_generated_qas.extend(cross_qas)
        
        # Format for evaluation
        evaluation_dataset = self.format_for_evaluation(all_generated_qas)
        
        # Add metadata
        metadata = {
            "generated_timestamp": datetime.now().isoformat(),
            "total_questions": len(evaluation_dataset),
            "source_chunks": len(chunks),
            "model_used": self.model,
            "questions_per_chunk": questions_per_chunk,
            "cross_document_questions": cross_doc_questions
        }
        
        # Save dataset
        final_data = {
            "metadata": metadata,
            "questions": evaluation_dataset
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {len(evaluation_dataset)} questions")
        logger.info(f"💾 Saved to: {output_file}")
        
        # Statistics
        stats = {
            "total_questions": len(evaluation_dataset),
            "by_difficulty": {},
            "by_category": {},
            "by_canton": {},
            "with_municipality": 0
        }
        
        for item in evaluation_dataset:
            # Difficulty stats
            diff = item.get('difficulty', 'unknown')
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
            
            # Category stats
            cat = item.get('category', 'unknown')
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            
            # Canton stats
            canton = item.get('canton')
            if canton:
                stats["by_canton"][canton] = stats["by_canton"].get(canton, 0) + 1
            
            # Municipality stats
            if item.get('municipality'):
                stats["with_municipality"] += 1
        
        logger.info("📊 Dataset Statistics:")
        logger.info(f"   By difficulty: {stats['by_difficulty']}")
        logger.info(f"   By category: {stats['by_category']}")
        logger.info(f"   By canton: {stats['by_canton']}")
        logger.info(f"   With municipality: {stats['with_municipality']}")
        
        return stats

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate ground truth using OpenAI")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--chunks-file", required=True, help="JSON file with document chunks")
    parser.add_argument("--output", default="ground_truth_dataset.json", help="Output file")
    parser.add_argument("--model", default="gpt-4-turbo-preview", help="OpenAI model to use")
    parser.add_argument("--questions-per-chunk", type=int, default=2, help="Questions per chunk")
    parser.add_argument("--cross-doc-questions", type=int, default=15, help="Cross-document questions")
    parser.add_argument("--enhance-existing", help="Enhance existing dataset file")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = OpenAIGroundTruthGenerator(args.api_key, args.model)
    
    # Generate dataset
    try:
        stats = await generator.generate_complete_dataset(
            chunks_file=args.chunks_file,
            output_file=args.output,
            questions_per_chunk=args.questions_per_chunk,
            cross_doc_questions=args.cross_doc_questions,
            enhance_existing=args.enhance_existing
        )
        
        print(f"\n✅ Successfully generated ground truth dataset!")
        print(f"📄 Output file: {args.output}")
        print(f"📊 Total questions: {stats['total_questions']}")
        
    except Exception as e:
        logger.error(f"❌ Error generating dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))