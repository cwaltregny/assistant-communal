import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

# Pour le reranking
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    RERANKER_AVAILABLE = True
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers non installé. Reranking et embeddings désactivés.")

# Pour la recherche hybride
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn non installé. Recherche hybride limitée.")

@dataclass
class SearchResult:
    """Résultat de recherche enrichi"""
    chunk_id: str
    content: str
    similarity_score: float
    source_file: str
    page_number: int
    document_type: str
    metadata: Dict
    # Nouveaux champs pour la recherche avancée
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    search_method: str = "semantic"

class HybridSearchEngine:
    """Moteur de recherche hybride combinant recherche sémantique et lexicale"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Vectorizer TF-IDF pour la recherche lexicale
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Modèle de reranking
        self.reranker = None
        self.load_reranker()
        
        # Paramètres de recherche hybride
        self.semantic_weight = config.get('semantic_weight', 0.7)
        self.lexical_weight = config.get('lexical_weight', 0.3)
        self.rerank_enabled = config.get('rerank_enabled', True)
        self.rerank_top_k = config.get('rerank_top_k', 20)
        
    def load_reranker(self):
        """Charge le modèle de reranking"""
        if not RERANKER_AVAILABLE or not self.config.get('rerank_enabled', True):
            return
            
        try:
            # Modèles de reranking recommandés (du plus léger au plus performant)
            reranker_models = [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Léger et rapide
                "cross-encoder/ms-marco-MiniLM-L-12-v2", # Équilibré
                "cross-encoder/ms-marco-TinyBERT-L-2-v2" # Très léger
            ]
            
            model_name = self.config.get('reranker_model', reranker_models[0])
            
            self.logger.info(f"Chargement du modèle de reranking: {model_name}")
            self.reranker = CrossEncoder(model_name, max_length=512)
            self.logger.info("Modèle de reranking chargé avec succès")
            
        except Exception as e:
            self.logger.warning(f"Impossible de charger le reranker: {e}")
            self.reranker = None
    
    def build_lexical_index(self, chunks: List[Dict]):
        """Construit l'index TF-IDF pour la recherche lexicale"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn non disponible pour la recherche lexicale")
            return
            
        try:
            # Prépare les textes pour TF-IDF
            texts = []
            for chunk in chunks:
                # Combine le contenu avec les métadonnées importantes
                text = chunk['content']
                
                # Ajoute les métadonnées pertinentes au texte pour améliorer la recherche
                metadata = chunk.get('metadata', {})
                if metadata.get('secteur'):
                    text += f" secteur {metadata['secteur']}"
                if metadata.get('canton'):
                    text += f" canton {metadata['canton']}"
                if chunk.get('document_type'):
                    text += f" type {chunk['document_type']}"
                    
                texts.append(text)
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='french',
                ngram_range=(1, 3),  
                min_df=2,           
                max_df=0.95,       
                sublinear_tf=True,  
                norm='l2'
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.logger.info(f"Index TF-IDF construit: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction de l'index TF-IDF: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def lexical_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Effectue une recherche lexicale avec TF-IDF"""
        if not self.tfidf_vectorizer or not SKLEARN_AVAILABLE:
            return []
            
        try:
            # Vectorise la requête
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calcule la similarité cosinus
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Récupère les top-k résultats
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Seulement les résultats pertinents
                    results.append((int(idx), float(similarities[idx])))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche lexicale: {e}")
            return []
    
    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Applique le reranking sur les résultats"""
        if not self.reranker or not RERANKER_AVAILABLE or len(results) == 0:
            return results
            
        try:
            # Prépare les paires (query, document) pour le reranking
            pairs = []
            for result in results:
                # Combine le contenu avec le contexte pour le reranking
                doc_text = result.content
                
                # Ajoute le contexte métadata pour améliorer le reranking
                if result.metadata.get('secteur'):
                    doc_text = f"Secteur {result.metadata['secteur']}: {doc_text}"
                if result.metadata.get('canton'):
                    doc_text = f"Canton {result.metadata['canton']}: {doc_text}"
                    
                pairs.append([query, doc_text])
            
            # Calcule les scores de reranking
            rerank_scores = self.reranker.predict(pairs)
            
            # Met à jour les résultats avec les scores de reranking
            for i, result in enumerate(results):
                result.rerank_score = float(rerank_scores[i])
                
                result.final_score = (
                    0.3 * result.similarity_score + 
                    0.7 * result.rerank_score       
                )
            
            # Trie par score final
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            self.logger.debug(f"Reranking appliqué sur {len(results)} résultats")
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors du reranking: {e}")
            return results

class EnhancedVectorStore:
    """Store vectoriel simplifié avec embeddings intégrés"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Composants vectoriels
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        # Modèle d'embeddings intégré
        self.embedding_model = None
        self.load_embedding_model()
        
        # Moteur de recherche hybride
        self.hybrid_engine = HybridSearchEngine(config)
        
        # Mapping pour les métadonnées
        self.canton_mapping = defaultdict(list)
        self.sector_mapping = defaultdict(list)
        self.document_type_mapping = defaultdict(list)
        
        # Paramètres de recherche
        self.hybrid_search_enabled = config.get('hybrid_search_enabled', True)
        self.expansion_enabled = config.get('query_expansion_enabled', True)
    
    def load_embedding_model(self):
        """Charge le modèle d'embeddings"""
        if not EMBEDDINGS_AVAILABLE:
            self.logger.error("sentence-transformers non disponible pour les embeddings")
            return
            
        try:
            embedding_models = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                "sentence-transformers/all-MiniLM-L6-v2",  # Très léger
                "sentence-transformers/all-mpnet-base-v2"   # Plus performant
            ]
            
            model_name = self.config.get('embedding_model', embedding_models[0])
            
            self.logger.info(f"Chargement du modèle d'embeddings: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("Modèle d'embeddings chargé avec succès")
            
        except Exception as e:
            self.logger.error(f"Impossible de charger le modèle d'embeddings: {e}")
            self.embedding_model = None
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Crée les embeddings pour une liste de textes"""
        if not self.embedding_model:
            self.logger.error("Modèle d'embeddings non disponible")
            return []
            
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des embeddings: {e}")
            return []
    
    def build_index(self, chunks: List[Dict]):
        """Construit l'index avec support de la recherche hybride"""
        self.logger.info("Construction de l'index vectoriel simplifié...")
        
        # Sauvegarde les chunks
        self.chunks = chunks
        
        # Génère les embeddings si nécessaire
        self._ensure_embeddings(chunks)
        
        # Construit l'index sémantique (FAISS)
        self._build_semantic_index(chunks)
        
        # Construit l'index lexical (TF-IDF)
        if self.hybrid_search_enabled:
            self.hybrid_engine.build_lexical_index(chunks)
        
        # Construit les mappings de métadonnées
        self._build_metadata_mappings(chunks)
        
        self.logger.info(f"Index construit avec {len(chunks)} chunks")
    
    def _ensure_embeddings(self, chunks: List[Dict]):
        """S'assure que tous les chunks ont des embeddings"""
        texts_to_embed = []
        indices_to_embed = []
        
        for i, chunk in enumerate(chunks):
            if 'embedding' not in chunk or chunk['embedding'] is None:
                texts_to_embed.append(chunk['content'])
                indices_to_embed.append(i)
        
        if texts_to_embed:
            self.logger.info(f"Génération d'embeddings pour {len(texts_to_embed)} chunks")
            embeddings = self.create_embeddings(texts_to_embed)
            
            for i, embedding in zip(indices_to_embed, embeddings):
                chunks[i]['embedding'] = embedding
    
    def _build_semantic_index(self, chunks: List[Dict]):
        """Construit l'index sémantique FAISS"""
        try:
            # Récupère les embeddings depuis les chunks
            embeddings = []
            for chunk in chunks:
                if 'embedding' in chunk and chunk['embedding'] is not None:
                    embeddings.append(chunk['embedding'])
                else:
                    self.logger.warning(f"Embedding manquant pour le chunk {chunk.get('chunk_id', 'unknown')}")
            
            if not embeddings:
                raise ValueError("Aucun embedding trouvé dans les chunks")
            
            embeddings = np.array(embeddings).astype('float32')
            self.embeddings = embeddings
            
            # Crée l'index FAISS
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product pour cosine similarity
            
            # Normalise les vecteurs pour la similarité cosinus
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            self.logger.info(f"Index FAISS créé avec {embeddings.shape[0]} vecteurs de dimension {dimension}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction de l'index sémantique: {e}")
            raise
    
    def _build_metadata_mappings(self, chunks: List[Dict]):
        """Construit les mappings de métadonnées"""
        self.canton_mapping.clear()
        self.sector_mapping.clear()
        self.document_type_mapping.clear()
        
        for i, chunk in enumerate(chunks):
            # Mapping par canton
            metadata = chunk.get('metadata', {})
            canton = metadata.get('canton')
            if canton:
                self.canton_mapping[canton.lower()].append(i)
            
            # Mapping par secteur
            secteur = metadata.get('secteur')
            if secteur:
                self.sector_mapping[secteur.lower()].append(i)
            
            # Mapping par type de document
            doc_type = chunk.get('document_type')
            if doc_type:
                self.document_type_mapping[doc_type.lower()].append(i)
    
    def expand_query(self, query: str) -> str:
        """Expansion de requête simple basée sur des synonymes"""
        if not self.expansion_enabled:
            return query
            
        # Dictionnaire de synonymes spécifiques au domaine PECC
        synonyms = {
            'transport': ['mobilité', 'déplacement', 'véhicule', 'circulation'],
            'bâtiment': ['construction', 'immobilier', 'habitat', 'logement'],
            'énergie': ['électricité', 'chauffage', 'renouvelable', 'efficacité'],
            'alimentation': ['nutrition', 'agriculture', 'bio', 'local'],
            'commune': ['municipalité', 'ville', 'localité'],
            'climat': ['climatique', 'environnement', 'CO2', 'carbone'],
            'mesure': ['action', 'politique', 'stratégie', 'initiative']
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for term, syns in synonyms.items():
            if term in query_lower:
                expanded_terms.extend(syns[:2])  # Ajoute max 2 synonymes
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query
    
    def search(self, query: str, k: int = 5, method: str = "hybrid", **kwargs) -> List[Dict]:
        """Recherche principale avec support hybride et reranking"""
        try:
            # Expansion de requête
            expanded_query = self.expand_query(query)
            
            if method == "hybrid" and self.hybrid_search_enabled:
                return self._hybrid_search(expanded_query, k, **kwargs)
            elif method == "semantic":
                return self._semantic_search(expanded_query, k, **kwargs)
            elif method == "lexical":
                return self._lexical_search(expanded_query, k, **kwargs)
            else:
                # Fallback sur la recherche sémantique
                return self._semantic_search(expanded_query, k, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    def _hybrid_search(self, query: str, k: int, **kwargs) -> List[Dict]:
        """Recherche hybride combinant sémantique et lexical"""
        try:
            # Paramètres de combinaison
            semantic_weight = kwargs.get('semantic_weight', self.hybrid_engine.semantic_weight)
            lexical_weight = kwargs.get('lexical_weight', self.hybrid_engine.lexical_weight)
            
            # Effectue la recherche sémantique
            semantic_results = self._semantic_search(query, k * 2, return_raw=True)
            
            # Effectue la recherche lexicale
            lexical_indices = self.hybrid_engine.lexical_search(query, k * 2)
            
            # Combine les résultats
            combined_scores = defaultdict(float)
            result_objects = {}
            
            # Ajoute les scores sémantiques
            for i, (idx, score) in enumerate(semantic_results):
                combined_scores[idx] += semantic_weight * score
                
                # Crée l'objet SearchResult
                chunk = self.chunks[idx]
                result_objects[idx] = SearchResult(
                    chunk_id=chunk['chunk_id'],
                    content=chunk['content'],
                    similarity_score=score,
                    source_file=chunk['source_file'],
                    page_number=chunk['page_number'],
                    document_type=chunk['document_type'],
                    metadata=chunk.get('metadata', {}),
                    semantic_score=score,
                    lexical_score=0.0,
                    search_method="hybrid"
                )
            
            # Ajoute les scores lexicaux
            for idx, score in lexical_indices:
                combined_scores[idx] += lexical_weight * score
                
                if idx in result_objects:
                    result_objects[idx].lexical_score = score
                else:
                    # Nouveau résultat trouvé seulement par recherche lexicale
                    chunk = self.chunks[idx]
                    result_objects[idx] = SearchResult(
                        chunk_id=chunk['chunk_id'],
                        content=chunk['content'],
                        similarity_score=score,
                        source_file=chunk['source_file'],
                        page_number=chunk['page_number'],
                        document_type=chunk['document_type'],
                        metadata=chunk.get('metadata', {}),
                        semantic_score=0.0,
                        lexical_score=score,
                        search_method="hybrid"
                    )
            
            # Met à jour les scores finaux
            for idx, combined_score in combined_scores.items():
                if idx in result_objects:
                    result_objects[idx].final_score = combined_score
            
            # Trie par score combiné et prend les top-k
            sorted_results = sorted(
                result_objects.values(),
                key=lambda x: x.final_score,
                reverse=True
            )[:self.hybrid_engine.rerank_top_k]
            
            # Applique le reranking si activé
            if self.hybrid_engine.rerank_enabled and len(sorted_results) > 1:
                sorted_results = self.hybrid_engine.rerank_results(query, sorted_results)
            
            # Retourne les k meilleurs résultats
            final_results = sorted_results[:k]
            
            # Convertit en format de retour attendu
            return [self._search_result_to_dict(result) for result in final_results]
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche hybride: {e}")
            # Fallback sur la recherche sémantique
            return self._semantic_search(query, k, **kwargs)
    
    def _semantic_search(self, query: str, k: int, return_raw: bool = False, **kwargs) -> Union[List[Dict], List[Tuple[int, float]]]:
        """Recherche sémantique avec FAISS"""
        try:
            if self.index is None:
                raise ValueError("Index non initialisé")
            
            # Génère l'embedding de la requête
            query_embeddings = self.create_embeddings([query])
            if not query_embeddings:
                self.logger.error("Impossible de générer l'embedding pour la requête")
                return []
            
            query_embedding = np.array([query_embeddings[0]]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Recherche dans l'index
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                # Vérifie que l'index est valide
                if idx >= len(self.chunks):
                    continue
                
                if return_raw:
                    results.append((idx, score))
                else:
                    chunk = self.chunks[idx]
                    result = SearchResult(
                        chunk_id=chunk['chunk_id'],
                        content=chunk['content'],
                        similarity_score=score,
                        source_file=chunk['source_file'],
                        page_number=chunk['page_number'],
                        document_type=chunk['document_type'],
                        metadata=chunk.get('metadata', {}),
                        semantic_score=score,
                        search_method="semantic"
                    )
                    results.append(self._search_result_to_dict(result))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche sémantique: {e}")
            return []
    
    def _lexical_search(self, query: str, k: int, **kwargs) -> List[Dict]:
        """Recherche lexicale pure avec TF-IDF"""
        try:
            lexical_results = self.hybrid_engine.lexical_search(query, k)
            
            results = []
            for idx, score in lexical_results:
                chunk = self.chunks[idx]
                result = SearchResult(
                    chunk_id=chunk['chunk_id'],
                    content=chunk['content'],
                    similarity_score=score,
                    source_file=chunk['source_file'],
                    page_number=chunk['page_number'],
                    document_type=chunk['document_type'],
                    metadata=chunk.get('metadata', {}),
                    lexical_score=score,
                    search_method="lexical"
                )
                results.append(self._search_result_to_dict(result))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche lexicale: {e}")
            return []
    
    def _search_result_to_dict(self, result: SearchResult) -> Dict:
        """Convertit un SearchResult en dictionnaire pour compatibilité"""
        return {
            'chunk_id': result.chunk_id,
            'content': result.content,
            'similarity_score': result.final_score if result.final_score > 0 else result.similarity_score,
            'source_file': result.source_file,
            'page_number': result.page_number,
            'document_type': result.document_type,
            'metadata': result.metadata,
            'semantic_score': result.semantic_score,
            'lexical_score': result.lexical_score,
            'rerank_score': result.rerank_score,
            'search_method': result.search_method
        }
    
    # ========== MÉTHODES DE COMPATIBILITÉ ==========
    
    def search_by_canton(self, query: str, canton: str, k: int = 5) -> List[Dict]:
        """Recherche filtrée par canton"""
        return self._filtered_search(query, k, canton_filter=canton.lower())
    
    def search_by_sector(self, query: str, sector: str, k: int = 5) -> List[Dict]:
        """Recherche filtrée par secteur"""
        return self._filtered_search(query, k, sector_filter=sector.lower())
    
    def search_by_document_type(self, query: str, doc_type: str, k: int = 5) -> List[Dict]:
        """Recherche filtrée par type de document"""
        return self._filtered_search(query, k, doc_type_filter=doc_type.lower())
    
    def _filtered_search(self, query: str, k: int, canton_filter: str = None, 
                        sector_filter: str = None, doc_type_filter: str = None) -> List[Dict]:
        """Recherche avec filtres de métadonnées"""
        try:
            # Détermine les indices candidats basés sur les filtres
            candidate_indices = set(range(len(self.chunks)))
            
            if canton_filter:
                canton_indices = self.canton_mapping.get(canton_filter, [])
                candidate_indices &= set(canton_indices)
            
            if sector_filter:
                sector_indices = self.sector_mapping.get(sector_filter, [])
                candidate_indices &= set(sector_indices)
            
            if doc_type_filter:
                doc_type_indices = self.document_type_mapping.get(doc_type_filter, [])
                candidate_indices &= set(doc_type_indices)
            
            if not candidate_indices:
                return []
            
            # Effectue la recherche sur les candidats
            all_results = self.search(query, k * 3, method="hybrid")
            
            # Filtre les résultats selon les critères
            filtered_results = []
            for result in all_results:
                chunk_idx = next((i for i, chunk in enumerate(self.chunks) 
                                if chunk['chunk_id'] == result['chunk_id']), None)
                
                if chunk_idx in candidate_indices:
                    filtered_results.append(result)
                    
                if len(filtered_results) >= k:
                    break
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche filtrée: {e}")
            return []
    
    def get_available_cantons(self) -> List[str]:
        """Retourne les cantons disponibles"""
        return list(self.canton_mapping.keys())
    
    def get_available_sectors(self) -> List[str]:
        """Retourne les secteurs disponibles"""
        return list(self.sector_mapping.keys())
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques de l'index"""
        stats = {
            'total_chunks': len(self.chunks),
            'index_type': 'Simplified Enhanced Hybrid FAISS',
            'available_cantons': len(self.canton_mapping),
            'available_sectors': len(self.sector_mapping),
            'document_types': len(self.document_type_mapping)
        }
        
        # Ajoute les statistiques de recherche avancée
        if self.hybrid_search_enabled:
            stats['hybrid_search'] = True
            stats['lexical_features'] = self.hybrid_engine.tfidf_matrix.shape[1] if self.hybrid_engine.tfidf_matrix is not None else 0
        
        if self.hybrid_engine.reranker:
            stats['reranking_enabled'] = True
            stats['reranker_model'] = getattr(self.hybrid_engine.reranker, 'model_name', 'unknown')
        
        return stats
    
    def save_index(self, path: str = None):
        """Sauvegarde l'index"""
        if path is None:
            path = self.config.get('index_path', 'vector_index')
        
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        try:
            # Sauvegarde l'index FAISS
            if self.index is not None:
                faiss.write_index(self.index, str(path / "faiss_index.bin"))
            
            # Sauvegarde les chunks et métadonnées
            with open(path / "chunks.pkl", "wb") as f:
                pickle.dump(self.chunks, f)
            
            # Sauvegarde les mappings
            mappings = {
                'canton_mapping': dict(self.canton_mapping),
                'sector_mapping': dict(self.sector_mapping),
                'document_type_mapping': dict(self.document_type_mapping)
            }
            
            with open(path / "mappings.pkl", "wb") as f:
                pickle.dump(mappings, f)
            
            # Sauvegarde l'index TF-IDF si disponible
            if self.hybrid_engine.tfidf_vectorizer is not None:
                with open(path / "tfidf_vectorizer.pkl", "wb") as f:
                    pickle.dump(self.hybrid_engine.tfidf_vectorizer, f)
                
                with open(path / "tfidf_matrix.pkl", "wb") as f:
                    pickle.dump(self.hybrid_engine.tfidf_matrix, f)
            
            # Sauvegarde la configuration
            config_to_save = {
                'semantic_weight': self.hybrid_engine.semantic_weight,
                'lexical_weight': self.hybrid_engine.lexical_weight,
                'hybrid_search_enabled': self.hybrid_search_enabled,
                'rerank_enabled': self.hybrid_engine.rerank_enabled
            }
            
            with open(path / "config.json", "w") as f:
                json.dump(config_to_save, f, indent=2)
            
            self.logger.info(f"Index sauvegardé dans {path}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def load_index(self, path: str = None) -> bool:
        """Charge l'index"""
        if path is None:
            path = self.config.get('index_path', 'vector_index')
        
        path = Path(path)
        
        if not path.exists():
            self.logger.warning(f"Chemin d'index non trouvé: {path}")
            return False
        
        try:
            # Charge l'index FAISS
            faiss_path = path / "faiss_index.bin"
            if faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))
            
            # Charge les chunks
            chunks_path = path / "chunks.pkl"
            if chunks_path.exists():
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
            
            # Charge les mappings
            mappings_path = path / "mappings.pkl"
            if mappings_path.exists():
                with open(mappings_path, "rb") as f:
                    mappings = pickle.load(f)
                    self.canton_mapping = defaultdict(list, mappings.get('canton_mapping', {}))
                    self.sector_mapping = defaultdict(list, mappings.get('sector_mapping', {}))
                    self.document_type_mapping = defaultdict(list, mappings.get('document_type_mapping', {}))
            
            # Charge l'index TF-IDF si disponible
            tfidf_vectorizer_path = path / "tfidf_vectorizer.pkl"
            tfidf_matrix_path = path / "tfidf_matrix.pkl"
            
            if tfidf_vectorizer_path.exists() and tfidf_matrix_path.exists():
                with open(tfidf_vectorizer_path, "rb") as f:
                    self.hybrid_engine.tfidf_vectorizer = pickle.load(f)
                
                with open(tfidf_matrix_path, "rb") as f:
                    self.hybrid_engine.tfidf_matrix = pickle.load(f)
            
            # Charge la configuration
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    saved_config = json.load(f)
                    
                    # Met à jour les paramètres
                    self.hybrid_engine.semantic_weight = saved_config.get('semantic_weight', 0.7)
                    self.hybrid_engine.lexical_weight = saved_config.get('lexical_weight', 0.3)
                    self.hybrid_search_enabled = saved_config.get('hybrid_search_enabled', True)
                    self.hybrid_engine.rerank_enabled = saved_config.get('rerank_enabled', True)
            
            self.logger.info(f"Index chargé depuis {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de l'index: {e}")
            return False