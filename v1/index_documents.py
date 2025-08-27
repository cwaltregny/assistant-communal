import os
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from process_documents import DocumentChunk

class VectorStore:
    def __init__(self, config):
        self.config = config
        self.embeddings_model = None
        self.index = None
        self.chunks_metadata = []
        self.vector_db_path = Path(config.VECTOR_DB_PATH)
        self.logger = logging.getLogger(__name__)
        
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self._load_embeddings_model()
        
    def _load_embeddings_model(self):
        """Charge le modèle d'embeddings multilingue"""
        try:
            self.embeddings_model = SentenceTransformer(self.config.EMBEDDINGS_MODEL)
            self.logger.info(f"Modèle d'embeddings chargé: {self.config.EMBEDDINGS_MODEL}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle d'embeddings: {e}")
            raise

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Crée les embeddings pour une liste de textes"""
        try:
            embeddings = self.embeddings_model.encode(
                texts, 
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des embeddings: {e}")
            raise

    def build_index(self, chunks: List[DocumentChunk]):
        """Construit l'index vectoriel FAISS"""
        self.logger.info("Construction de l'index vectoriel...")
        
        texts = [chunk.content for chunk in chunks]
        
        embeddings = self.create_embeddings(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        
        # Sauvegarde les métadonnées
        self.chunks_metadata = [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'source_file': chunk.source_file,
                'page_number': chunk.page_number,
                'document_type': chunk.document_type,
                'metadata': chunk.metadata
            }
            for chunk in chunks
        ]
        
        self.logger.info(f"Index construit avec {len(chunks)} chunks")

    def save_index(self):
        """Sauvegarde l'index et les métadonnées"""
        try:
            index_path = self.vector_db_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            metadata_path = self.vector_db_path / "chunks_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Index et métadonnées sauvegardés")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise

    def load_index(self) -> bool:
        """Charge l'index et les métadonnées depuis le disque"""
        try:
            index_path = self.vector_db_path / "faiss_index.bin"
            metadata_path = self.vector_db_path / "chunks_metadata.json"
            
            if not (index_path.exists() and metadata_path.exists()):
                self.logger.warning("Index ou métadonnées non trouvés")
                return False
            
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            
            self.logger.info(f"Index chargé avec {len(self.chunks_metadata)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {e}")
            return False

    def search(self, query: str, k: int = 5, filter_by: Optional[Dict] = None) -> List[Dict]:
        """Recherche par similarité vectorielle"""
        try:
            if self.index is None:
                raise ValueError("Index non initialisé")
            
            query_embedding = self.create_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Recherche les k plus proches voisins
            scores, indices = self.index.search(query_embedding, min(k * 2, len(self.chunks_metadata)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks_metadata):
                    chunk_metadata = self.chunks_metadata[idx]
                    
                    # Applique les filtres si spécifiés
                    if filter_by and not self._matches_filter(chunk_metadata, filter_by):
                        continue
                    
                    result = {
                        'content': chunk_metadata['content'],
                        'source_file': chunk_metadata['source_file'],
                        'page_number': chunk_metadata['page_number'],
                        'document_type': chunk_metadata['document_type'],
                        'similarity_score': float(score),
                        'metadata': chunk_metadata['metadata'],
                        'chunk_id': chunk_metadata['chunk_id']
                    }
                    results.append(result)
                    
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche: {e}")
            return []

    def _matches_filter(self, chunk_metadata: Dict, filters: Dict) -> bool:
        """Vérifie si un chunk correspond aux filtres"""
        for key, value in filters.items():
            if key == 'document_type':
                if chunk_metadata.get('document_type') != value:
                    return False
            elif key == 'canton':
                if chunk_metadata.get('metadata', {}).get('canton') != value:
                    return False
            elif key == 'secteur':
                if chunk_metadata.get('metadata', {}).get('secteur') != value:
                    return False
        return True

    def search_by_document_type(self, query: str, document_type: str, k: int = 5) -> List[Dict]:
        """Recherche filtrée par type de document"""
        return self.search(query, k, filter_by={'document_type': document_type})

    def search_by_canton(self, query: str, canton: str, k: int = 5) -> List[Dict]:
        """Recherche filtrée par canton"""
        return self.search(query, k, filter_by={'canton': canton})

    def search_by_sector(self, query: str, secteur: str, k: int = 5) -> List[Dict]:
        """Recherche filtrée par secteur"""
        return self.search(query, k, filter_by={'secteur': secteur})

    def get_available_cantons(self) -> List[str]:
        """Retourne la liste des cantons disponibles"""
        cantons = set()
        for chunk in self.chunks_metadata:
            canton = chunk.get('metadata', {}).get('canton')
            if canton and canton != 'Canton_Inconnu':
                cantons.add(canton)
        return sorted(list(cantons))

    def get_available_sectors(self) -> List[str]:
        """Retourne la liste des secteurs disponibles"""
        secteurs = set()
        for chunk in self.chunks_metadata:
            secteur = chunk.get('metadata', {}).get('secteur')
            if secteur and secteur != 'Secteur_Inconnu':
                secteurs.add(secteur)
        return sorted(list(secteurs))

    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur la base de données"""
        if not self.chunks_metadata:
            return {}
        
        stats = {
            'total_chunks': len(self.chunks_metadata),
            'documents_by_type': {},
            'documents_by_canton': {},
            'documents_by_sector': {}
        }
        
        for chunk in self.chunks_metadata:
            doc_type = chunk.get('document_type', 'unknown')
            stats['documents_by_type'][doc_type] = stats['documents_by_type'].get(doc_type, 0) + 1
            
            canton = chunk.get('metadata', {}).get('canton', 'unknown')
            stats['documents_by_canton'][canton] = stats['documents_by_canton'].get(canton, 0) + 1
            
            secteur = chunk.get('metadata', {}).get('secteur', 'unknown')
            stats['documents_by_sector'][secteur] = stats['documents_by_sector'].get(secteur, 0) + 1
        
        return stats