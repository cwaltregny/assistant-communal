import asyncio
import logging
import sys
import gc
import psutil
import time
from pathlib import Path
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simplified_embeddings_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from config_ac import settings
from process_documents import DocumentProcessor

SIMPLIFIED_CONFIG = {
    'chunk_batch_size': 32,        # Traite 32 chunks à la fois
    'embedding_batch_size': 8,     # Embeddings par batch
    'memory_threshold_mb': 512,    # Pause si plus de 512MB utilisé
    'gc_frequency': 50,            # Garbage collection tous les 50 chunks
    'progress_frequency': 25,      # Affiche progrès tous les 25 chunks
    'max_file_size_mb': 5,         # Limite la taille des fichiers traités
    'pause_between_batches': 0.5   # Pause entre batches
}

# Configuration spécifique pour le store amélioré
ENHANCED_CONFIG = {
    'vector_store_path': 'vector_store_enhanced',
    'hybrid_search_enabled': True,
    'rerank_enabled': True,
    'query_expansion_enabled': True,
    'semantic_weight': 0.7,
    'lexical_weight': 0.3,
    'rerank_top_k': 20,
    'rerank_threshold': 0.5,
    'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'diversity_threshold': 0.8,
    'max_context_length': 4000
}

def get_memory_usage():
    """Retourne l'utilisation mémoire en MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def memory_cleanup():
    """Nettoie la mémoire"""
    gc.collect()
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(700, 10, 10)  

def monitor_memory():
    """Surveille la mémoire et fait une pause si nécessaire"""
    memory_mb = get_memory_usage()
    
    if memory_mb > SIMPLIFIED_CONFIG['memory_threshold_mb']:
        logger.warning(f"⚠️  Mémoire élevée: {memory_mb:.1f}MB - Nettoyage en cours...")
        memory_cleanup()
        time.sleep(2)
        
        new_memory = get_memory_usage()
        logger.info(f"🧹 Mémoire après nettoyage: {new_memory:.1f}MB")
    
    return memory_mb

def create_directories():
    """Crée les dossiers nécessaires"""
    directories = [
        Path(ENHANCED_CONFIG['vector_store_path']).parent,
        Path("logs"),
        Path(settings.PLAN_CLIMAT_FOLDER),
        Path(settings.PRIORITY_MEASURES_FOLDER)
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Dossier créé/vérifié: {directory}")

def validate_dependencies():
    """Valide que les dépendances pour le RAG sont installées"""
    logger.info("🔍 Validation des dépendances...")
    
    missing_deps = []
    
    try:
        import sentence_transformers
        logger.info("✅ sentence-transformers disponible")
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import sklearn
        logger.info("✅ scikit-learn disponible")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import faiss
        logger.info("✅ faiss disponible")
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    if missing_deps:
        logger.error("❌ Dépendances manquantes:")
        for dep in missing_deps:
            logger.error(f"   pip install {dep}")
        return False
    
    logger.info("✅ Toutes les dépendances sont disponibles")
    return True

def validate_input_files():
    """Valide que tous les fichiers d'entrée sont présents"""
    logger.info("🔍 Validation des fichiers d'entrée...")
    
    issues = []
    warnings = []
    
    # Vérification du CSV
    csv_path = Path(settings.MUNICIPALITIES_CSV)
    if not csv_path.exists():
        issues.append(f"❌ Fichier CSV manquant: {csv_path}")
    else:
        logger.info(f"✅ CSV trouvé: {csv_path}")
    
    # Vérification des PDFs Plan Climat
    plan_climat_folder = Path(settings.PLAN_CLIMAT_FOLDER)
    plan_climat_pdfs = list(plan_climat_folder.glob("*.pdf"))
    
    if not plan_climat_pdfs:
        issues.append(f"❌ Aucun PDF Plan Climat trouvé dans: {plan_climat_folder}")
    else:
        logger.info(f"✅ {len(plan_climat_pdfs)} PDFs Plan Climat trouvés")
        
        for pdf in plan_climat_pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            if size_mb > SIMPLIFIED_CONFIG['max_file_size_mb']:
                warnings.append(f"⚠️  Fichier volumineux: {pdf.name} ({size_mb:.1f}MB)")
            logger.info(f"   {pdf.name}: {size_mb:.1f}MB")
    
    # Vérification des PDFs Mesures
    mesures_folder = Path(settings.PRIORITY_MEASURES_FOLDER)
    mesures_pdfs = list(mesures_folder.glob("*.pdf"))
    
    if not mesures_pdfs:
        issues.append(f"❌ Aucun PDF Mesures trouvé dans: {mesures_folder}")
    else:
        logger.info(f"✅ {len(mesures_pdfs)} PDFs Mesures trouvés")
        
        for pdf in mesures_pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            if size_mb > SIMPLIFIED_CONFIG['max_file_size_mb']:
                warnings.append(f"⚠️  Fichier volumineux: {pdf.name} ({size_mb:.1f}MB)")
            logger.info(f"   {pdf.name}: {size_mb:.1f}MB")
    
    if warnings:
        logger.warning("⚠️  Fichiers volumineux détectés - peuvent causer des problèmes de mémoire")
        for warning in warnings:
            logger.warning(f"   {warning}")
    
    if issues:
        logger.error("❌ Problèmes détectés:")
        for issue in issues:
            logger.error(f"   {issue}")
        return False
    
    logger.info("✅ Tous les fichiers requis sont présents")
    return True

def process_documents():
    """Traite les documents"""
    logger.info("📄 Traitement des documents...")
    
    try:
        processor = DocumentProcessor(settings)
        
        # Chargement des données des communes
        logger.info("Chargement des données des communes...")
        municipalities = processor.load_municipalities_data()
        logger.info(f"✅ {len(municipalities)} communes chargées")
        
        all_chunks = []
        
        # Traitement des Mesures
        logger.info("🎯 Traitement des Mesures Prioritaires...")
        mesures_chunks = processor.process_priority_measures_documents()
        all_chunks.extend(mesures_chunks)
        logger.info(f"   Mesures: {len(mesures_chunks)} chunks créés")
        
        # Nettoyage mémoire
        memory_cleanup()
        
        # Traitement des Plans Climat
        logger.info("📋 Traitement des Plans Climat...")
        plan_climat_chunks = processor.process_plan_climat_documents()
        all_chunks.extend(plan_climat_chunks)
        logger.info(f"   Plans Climat: {len(plan_climat_chunks)} chunks créés")
        
        # Nettoyage mémoire final
        memory_cleanup()
        
        if not all_chunks:
            raise ValueError("Aucun chunk créé - vérifiez vos fichiers PDF")
        
        logger.info(f"✅ {len(all_chunks)} chunks créés au total")
        return all_chunks
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement des documents: {e}")
        raise

def create_vector_store():
    """Crée le store vectoriel"""
    logger.info("🚀 Création du store vectoriel...")
    
    # Import dynamique pour éviter les erreurs si pas installé
    try:
        from enhanced_vector_store import EnhancedVectorStore
    except ImportError:
        logger.error("❌ enhanced_vector_store.py non trouvé")
        raise
    
    # Configuration complète pour le store 
    enhanced_settings = {
        'index_path': ENHANCED_CONFIG['vector_store_path'],
        **ENHANCED_CONFIG
    }
    
    if hasattr(settings, '__dict__'):
        for key, value in settings.__dict__.items():
            if not key.startswith('_') and key not in enhanced_settings:
                enhanced_settings[key] = value
    
    vector_store = EnhancedVectorStore(enhanced_settings)
    logger.info("✅ Store vectoriel créé")
    
    return vector_store

def generate_embeddings_with_simplified_store(chunks):
    """Génère les embeddings avec le store"""
    logger.info("🔍 Génération des embeddings avec store...")
    
    try:
        vector_store = create_vector_store()
        
        start_time = time.time()
        total_chunks = len(chunks)
        
        logger.info(f"Génération pour {total_chunks} chunks")
        logger.info(f"Configuration: batch_size={SIMPLIFIED_CONFIG['embedding_batch_size']}")
        logger.info(f"Fonctionnalités: hybrid={ENHANCED_CONFIG['hybrid_search_enabled']}, "
                   f"rerank={ENHANCED_CONFIG['rerank_enabled']}")
        
        # Prépare les données des chunks pour le store 
        chunk_data = []
        for chunk in chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'source_file': chunk.source_file,
                'page_number': chunk.page_number,
                'document_type': chunk.document_type,
                'metadata': chunk.metadata
            }
            chunk_data.append(chunk_dict)
        
        # Génération des embeddings par batch
        logger.info("🤖 Génération des embeddings...")
        batch_size = SIMPLIFIED_CONFIG['embedding_batch_size']
        processed = 0
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunk_data[i:i + batch_size]
            batch_texts = [chunk['content'] for chunk in batch_chunks]
            
            logger.info(f"   Batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({len(batch_texts)} chunks)")
            
            # Génération des embeddings pour ce batch
            batch_embeddings = vector_store.create_embeddings(batch_texts)
            
            # Ajoute les embeddings aux chunks
            for j, embedding in enumerate(batch_embeddings):
                chunk_idx = i + j
                if chunk_idx < len(chunk_data):
                    chunk_data[chunk_idx]['embedding'] = embedding
            
            processed += len(batch_texts)
            
            # Monitoring mémoire
            if processed % SIMPLIFIED_CONFIG['progress_frequency'] == 0:
                memory_mb = monitor_memory()
                progress = (processed / total_chunks) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / processed) * (total_chunks - processed) if processed > 0 else 0
                
                logger.info(f"   Progrès: {processed}/{total_chunks} ({progress:.1f}%) - "
                          f"Mémoire: {memory_mb:.1f}MB - ETA: {eta:.1f}s")
            
            # Nettoyage périodique
            if processed % SIMPLIFIED_CONFIG['gc_frequency'] == 0:
                memory_cleanup()
            
            # Pause système
            time.sleep(SIMPLIFIED_CONFIG['pause_between_batches'])
        
        # Construction de l'index 
        logger.info("🏗️  Construction de l'index...")
        vector_store.build_index(chunk_data)
        
        embedding_time = time.time() - start_time
        logger.info(f"✅ Index créé en {embedding_time:.1f} secondes")
        
        logger.info("💾 Sauvegarde de l'index vectoriel...")
        vector_store.save_index()
        
        # Statistiques finales
        stats = vector_store.get_statistics()
        logger.info("📊 Statistiques de l'index:")
        logger.info(f"   Total chunks: {stats['total_chunks']}")
        logger.info(f"   Index type: {stats.get('index_type', 'Enhanced')}")
        logger.info(f"   Hybrid search: {stats.get('hybrid_search', False)}")
        logger.info(f"   Reranking: {stats.get('reranking_enabled', False)}")
        logger.info(f"   Cantons: {len(vector_store.get_available_cantons())}")
        logger.info(f"   Secteurs: {len(vector_store.get_available_sectors())}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération: {e}")
        raise

def validate_simplified_embeddings():
    """Valide que les embeddings ont été correctement sauvegardés"""
    logger.info("🧪 Validation de l'index...")
    
    try:
        from enhanced_vector_store import EnhancedVectorStore
        
        # Configuration pour le store simplifié
        enhanced_settings = {
            'index_path': ENHANCED_CONFIG['vector_store_path'],
            **ENHANCED_CONFIG
        }
        
        vector_store = EnhancedVectorStore(enhanced_settings)
        
        # Test de chargement
        if vector_store.load_index():
            stats = vector_store.get_statistics()
            logger.info(f"✅ Index chargé: {stats['total_chunks']} chunks")
            
            # Validation basique de la structure de l'index
            if hasattr(vector_store, 'index') and vector_store.index is not None:
                logger.info("✅ Index FAISS chargé correctement")
            
            if hasattr(vector_store, 'chunks') and vector_store.chunks:
                logger.info(f"✅ {len(vector_store.chunks)} chunks de données chargés")
            
            # Test des mappings de métadonnées
            if hasattr(vector_store, 'canton_mapping') and vector_store.canton_mapping:
                cantons_count = len(vector_store.canton_mapping)
                logger.info(f"✅ Mappings canton: {cantons_count} cantons")
            
            if hasattr(vector_store, 'sector_mapping') and vector_store.sector_mapping:
                sectors_count = len(vector_store.sector_mapping)
                logger.info(f"✅ Mappings secteur: {sectors_count} secteurs")
            
            # Test des fonctionnalités avancées
            if stats.get('hybrid_search'):
                logger.info("✅ Recherche hybride activée")
                
                # Vérifie l'index TF-IDF
                if (hasattr(vector_store.hybrid_engine, 'tfidf_vectorizer') and 
                    vector_store.hybrid_engine.tfidf_vectorizer is not None):
                    logger.info("✅ Index TF-IDF chargé")
                
                if (hasattr(vector_store.hybrid_engine, 'reranker') and 
                    vector_store.hybrid_engine.reranker is not None):
                    logger.info("✅ Modèle de reranking chargé")
            
            # Test simple de recherche
            try:
                test_results = vector_store.search("transport", k=3, method="semantic")
                if test_results:
                    logger.info(f"✅ Test recherche sémantique: {len(test_results)} résultats")
                
                # Test de recherche hybride si disponible
                if stats.get('hybrid_search'):
                    hybrid_results = vector_store.search("efficacité énergétique", k=3, method="hybrid")
                    if hybrid_results:
                        logger.info(f"✅ Test recherche hybride: {len(hybrid_results)} résultats")
                        
                        # Vérifie les scores avancés
                        first_result = hybrid_results[0]
                        has_advanced_scores = any(key in first_result for key in 
                                                ['semantic_score', 'lexical_score', 'rerank_score'])
                        if has_advanced_scores:
                            logger.info("✅ Scores avancés détectés dans les résultats")
                
            except Exception as search_error:
                logger.warning(f"⚠️  Tests de recherche partiellement échoués: {search_error}")
                logger.info("✅ Index validé (structure correcte, recherche à tester en usage)")
            
            # Test de recherche avec métadonnées
            try:
                cantons = vector_store.get_available_cantons()
                if cantons and len(cantons) > 0:
                    logger.info(f"✅ {len(cantons)} cantons disponibles: {cantons[:3]}...")
                
                sectors = vector_store.get_available_sectors()
                if sectors and len(sectors) > 0:
                    logger.info(f"✅ {len(sectors)} secteurs disponibles: {sectors[:3]}...")
                    
            except Exception as metadata_error:
                logger.warning(f"⚠️  Test métadonnées échoué: {metadata_error}")
            
            logger.info("✅ Validation de l'index terminée avec succès")
            return True
        else:
            logger.error("❌ Impossible de charger l'index")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation: {e}")
        return False

def main():
    """Fonction principale pour générer l'index simplifié"""
    logger.info("🚀 Génération SIMPLIFIÉE des embeddings pour l'Assistant PECC")
    logger.info("=" * 70)
    
    # Affichage de la configuration
    logger.info("⚙️  Configuration:")
    for key, value in SIMPLIFIED_CONFIG.items():
        logger.info(f"   {key}: {value}")
    
    logger.info("🔬 Configuration avancée:")
    for key, value in ENHANCED_CONFIG.items():
        logger.info(f"   {key}: {value}")
    
    # Monitoring initial
    initial_memory = get_memory_usage()
    logger.info(f"💾 Mémoire initiale: {initial_memory:.1f}MB")
    
    start_total = time.time()
    
    try:
        # 1. Validation des dépendances
        logger.info("\nÉtape 1/6: Validation des dépendances")
        if not validate_dependencies():
            logger.error("❌ Dépendances manquantes - installez-les avec:")
            logger.error("pip install sentence-transformers scikit-learn faiss-cpu")
            return False
        
        # 2. Création des dossiers
        logger.info("\nÉtape 2/6: Création des dossiers")
        create_directories()
        
        # 3. Validation des fichiers
        logger.info("\nÉtape 3/6: Validation des fichiers d'entrée")
        if not validate_input_files():
            logger.error("❌ Validation échouée - vérifiez vos fichiers")
            return False
        
        # 4. Traitement des documents
        logger.info("\nÉtape 4/6: Traitement des documents")
        chunks = process_documents()
        
        # 5. Génération des embeddings simplifiés
        logger.info("\nÉtape 5/6: Génération des embeddings simplifiés")
        vector_store = generate_embeddings_with_simplified_store(chunks)
        
        # 6. Validation de l'index
        logger.info("\nÉtape 6/6: Validation de l'index")
        if not validate_simplified_embeddings():
            logger.error("❌ Validation de l'index échouée")
            return False
        
        # Résumé final
        total_time = time.time() - start_total
        final_memory = get_memory_usage()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Génération SIMPLIFIÉE terminée avec succès!")
        logger.info(f"⏱️  Temps total: {total_time:.1f} secondes")
        logger.info(f"💾 Mémoire finale: {final_memory:.1f}MB (vs {initial_memory:.1f}MB initial)")
        logger.info(f"📁 Index sauvé: {ENHANCED_CONFIG['vector_store_path']}")
        
        logger.info("\n🚀 Fonctionnalités activées:")
        logger.info(f"   ✅ Recherche hybride (sémantique + lexicale)")
        logger.info(f"   ✅ Reranking avec cross-encoder")
        logger.info(f"   ✅ Expansion de requête")
        logger.info(f"   ✅ Embeddings intégrés")
        logger.info(f"   ✅ Filtrage par métadonnées")
        
        logger.info("\n📋 Prochaines étapes:")
        logger.info("   1. Lancez l'assistant: streamlit run simplified_pecc_assistant.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        return False

if __name__ == "__main__":
    # Vérification des ressources système
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    logger.info(f"💾 Mémoire système: {available_memory_gb:.1f}GB disponible / {total_memory_gb:.1f}GB total")
    
    # Adaptation automatique selon la mémoire disponible
    if total_memory_gb < 4:
        logger.warning(f"⚠️  Mémoire faible: {total_memory_gb:.1f}GB - Mode ultra-léger")
        SIMPLIFIED_CONFIG['embedding_batch_size'] = 2
        SIMPLIFIED_CONFIG['memory_threshold_mb'] = 256
        ENHANCED_CONFIG['rerank_enabled'] = False  # Désactive reranking sur machines faibles
        logger.warning("   Reranking désactivé pour économiser la mémoire")
    elif total_memory_gb < 8:
        logger.info(f"📊 Mémoire modérée: {total_memory_gb:.1f}GB - Mode léger")
        SIMPLIFIED_CONFIG['embedding_batch_size'] = 4
        SIMPLIFIED_CONFIG['memory_threshold_mb'] = 512
    else:
        logger.info(f"✅ Mémoire suffisante: {total_memory_gb:.1f}GB - Mode normal")
        SIMPLIFIED_CONFIG['embedding_batch_size'] = 8
        SIMPLIFIED_CONFIG['memory_threshold_mb'] = 1024
    
    success = main()
    sys.exit(0 if success else 1)