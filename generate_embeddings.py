import asyncio
import logging
import sys
import gc
import psutil
import time
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embeddings_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from config_ac import settings
from process_documents import DocumentProcessor
from index_documents import VectorStore

# Configuration ultra-légère pour machines limitées
LIGHTWEIGHT_CONFIG = {
    'chunk_batch_size': 32,        # Traite 16 chunks à la fois (réduit)
    'embedding_batch_size': 8,     # Embeddings par batch (très petit)
    'memory_threshold_mb': 512,    # Pause si plus de 512MB utilisé (réduit)
    'gc_frequency': 50,            # Garbage collection tous les 50 chunks (plus fréquent)
    'progress_frequency': 25,      # Affiche progrès tous les 25 chunks
    'max_file_size_mb': 5,         # Limite la taille des fichiers traités
    'pause_between_batches': 0.5   # Pause plus longue entre batches
}

def get_memory_usage():
    """Retourne l'utilisation mémoire en MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def memory_cleanup():
    """Nettoie la mémoire"""
    gc.collect()
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(700, 10, 10)  # Plus agressif

def monitor_memory():
    """Surveille la mémoire et fait une pause si nécessaire"""
    memory_mb = get_memory_usage()
    
    if memory_mb > LIGHTWEIGHT_CONFIG['memory_threshold_mb']:
        logger.warning(f"⚠️  Mémoire élevée: {memory_mb:.1f}MB - Nettoyage en cours...")
        memory_cleanup()
        time.sleep(2)  # Pause pour laisser le système respirer
        
        # Vérification après nettoyage
        new_memory = get_memory_usage()
        logger.info(f"🧹 Mémoire après nettoyage: {new_memory:.1f}MB")
    
    return memory_mb

def create_directories():
    """Crée les dossiers nécessaires"""
    directories = [
        Path(settings.VECTOR_DB_PATH),
        Path("logs"),
        Path(settings.PLAN_CLIMAT_FOLDER),
        Path(settings.PRIORITY_MEASURES_FOLDER)
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Dossier créé/vérifié: {directory}")

def validate_input_files():
    """Valide que tous les fichiers d'entrée sont présents et pas trop volumineux"""
    logger.info("🔍 Validation des fichiers d'entrée...")
    
    issues = []
    warnings = []
    
    # Vérification du CSV
    csv_path = Path(settings.MUNICIPALITIES_CSV)
    if not csv_path.exists():
        issues.append(f"❌ Fichier CSV manquant: {csv_path}")
    else:
        logger.info(f"✅ CSV trouvé: {csv_path}")
    
    # Vérification des PDFs Plan Climat avec taille
    plan_climat_folder = Path(settings.PLAN_CLIMAT_FOLDER)
    plan_climat_pdfs = list(plan_climat_folder.glob("*.pdf"))
    expected_pcc = ['pcc_fribourg.pdf', 'pcc_geneve.pdf', 'pcc_neuchatel.pdf', 'pcc_valais.pdf', 'pcc_vaud.pdf']
    
    if not plan_climat_pdfs:
        issues.append(f"❌ Aucun PDF Plan Climat trouvé dans: {plan_climat_folder}")
    else:
        logger.info(f"✅ {len(plan_climat_pdfs)} PDFs Plan Climat trouvés")
        
        # Vérification de la taille des fichiers
        for pdf in plan_climat_pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            if size_mb > LIGHTWEIGHT_CONFIG['max_file_size_mb']:
                warnings.append(f"⚠️  Fichier volumineux: {pdf.name} ({size_mb:.1f}MB)")
            logger.info(f"   {pdf.name}: {size_mb:.1f}MB")
        
        found_names = [pdf.name for pdf in plan_climat_pdfs]
        missing_pcc = [f for f in expected_pcc if f not in found_names]
        if missing_pcc:
            logger.warning(f"⚠️  PDFs Plan Climat manquants: {missing_pcc}")
    
    # Vérification des PDFs Mesures avec taille
    mesures_folder = Path(settings.PRIORITY_MEASURES_FOLDER)
    mesures_pdfs = list(mesures_folder.glob("*.pdf"))
    expected_mesures = [
        'Bons_et_moins_bons_exemples_mobilite.pdf',
        'Fiche_Alimentation_durable.pdf',
        'Fiche_Amenagement_territoire.pdf',
        'Fiche_Construction_bois.pdf',
        'Fiche_Energie.pdf',
        'Fiche_Modes_transport_alternatifs.pdf',
        'Fiche_Renovation_energetique_batiments.pdf',
        'STC-note-decideurs.pdf'
    ]
    
    if not mesures_pdfs:
        issues.append(f"❌ Aucun PDF Mesures trouvé dans: {mesures_folder}")
    else:
        logger.info(f"✅ {len(mesures_pdfs)} PDFs Mesures trouvés")
        
        # Vérification de la taille des fichiers
        for pdf in mesures_pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            if size_mb > LIGHTWEIGHT_CONFIG['max_file_size_mb']:
                warnings.append(f"⚠️  Fichier volumineux: {pdf.name} ({size_mb:.1f}MB)")
            logger.info(f"   {pdf.name}: {size_mb:.1f}MB")
        
        found_names = [pdf.name for pdf in mesures_pdfs]
        missing_mesures = [f for f in expected_mesures if f not in found_names]
        if missing_mesures:
            logger.warning(f"⚠️  PDFs Mesures manquants: {missing_mesures}")
    
    # Affichage des warnings
    if warnings:
        logger.warning("⚠️  Fichiers volumineux détectés:")
        for warning in warnings:
            logger.warning(f"   {warning}")
        logger.warning(f"   Les fichiers >5MB peuvent causer des problèmes de mémoire")
    
    if issues:
        logger.error("❌ Problèmes détectés:")
        for issue in issues:
            logger.error(f"   {issue}")
        return False
    
    logger.info("✅ Tous les fichiers requis sont présents")
    return True

def process_documents_lightweight():
    """Traite les documents de façon allégée"""
    logger.info("📄 Traitement léger des documents...")
    
    try:
        processor = DocumentProcessor(settings)
        
        # Chargement des données des communes
        logger.info("Chargement des données des communes...")
        municipalities = processor.load_municipalities_data()
        logger.info(f"✅ {len(municipalities)} communes chargées")
        
        # Traitement par batch des documents
        logger.info("Traitement des documents par batch...")
        all_chunks = []
        
        # Traitement des Mesures
        logger.info("🎯 Traitement des Mesures Prioritaires...")
        mesures_chunks = processor.process_priority_measures_documents()
        all_chunks.extend(mesures_chunks)
        
        # Traitement des Plans Climat
        logger.info("📋 Traitement des Plans Climat...")
        plan_climat_chunks = processor.process_plan_climat_documents()
        all_chunks.extend(plan_climat_chunks)
        
        # Nettoyage mémoire après Plans Climat
        memory_cleanup()
        logger.info(f"   Plans Climat: {len(plan_climat_chunks)} chunks créés")
        
        # Nettoyage mémoire final
        memory_cleanup()
        
        if not all_chunks:
            raise ValueError("Aucun chunk créé - vérifiez vos fichiers PDF")
        
        logger.info(f"✅ {len(all_chunks)} chunks créés au total:")
        logger.info(f"   📋 Plan Climat: {len(plan_climat_chunks)} chunks")
        logger.info(f"   🎯 Mesures: {len(mesures_chunks)} chunks")
        
        return all_chunks
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement des documents: {e}")
        raise

def generate_embeddings_lightweight(chunks):
    """Génère les embeddings de façon allégée"""
    logger.info("🔍 Génération légère des embeddings...")
    
    try:
        vector_store = VectorStore(settings)
        
        start_time = time.time()
        total_chunks = len(chunks)
        
        logger.info(f"Génération des embeddings pour {total_chunks} chunks en mode léger...")
        logger.info(f"Configuration: batch_size={LIGHTWEIGHT_CONFIG['embedding_batch_size']}")
    
        
        embeddings_list = []
        processed = 0
        
        batch_size = LIGHTWEIGHT_CONFIG['embedding_batch_size']
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            # Génération des embeddings pour ce batch
            logger.info(f"   Traitement batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({len(batch_texts)} chunks)")
            
            batch_embeddings = vector_store.create_embeddings(batch_texts)
            embeddings_list.append(batch_embeddings)
            
            processed += len(batch_texts)
            
            # Monitoring et nettoyage
            if processed % LIGHTWEIGHT_CONFIG['progress_frequency'] == 0:
                memory_mb = monitor_memory()
                progress = (processed / total_chunks) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / processed) * (total_chunks - processed) if processed > 0 else 0
                
                logger.info(f"   Progrès: {processed}/{total_chunks} ({progress:.1f}%) - "
                          f"Mémoire: {memory_mb:.1f}MB - ETA: {eta:.1f}s")
            
            # Nettoyage périodique
            if processed % LIGHTWEIGHT_CONFIG['gc_frequency'] == 0:
                memory_cleanup()
            
            # Petite pause pour laisser respirer le système
            time.sleep(LIGHTWEIGHT_CONFIG['pause_between_batches'])
        
        # Assemblage final des embeddings
        logger.info("🔧 Assemblage final des embeddings...")
        import numpy as np
        all_embeddings = np.vstack(embeddings_list)
        
        # Construction de l'index FAISS
        logger.info("🏗️  Construction de l'index FAISS...")
        import faiss
        
        dimension = all_embeddings.shape[1]
        vector_store.index = faiss.IndexFlatIP(dimension)
        
        # Normalisation et ajout à l'index
        faiss.normalize_L2(all_embeddings)
        vector_store.index.add(all_embeddings)
        
        # Sauvegarde des métadonnées
        vector_store.chunks_metadata = [
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
        
        embedding_time = time.time() - start_time
        logger.info(f"✅ Embeddings créés en {embedding_time:.1f} secondes")
        
        # Sauvegarde
        logger.info("💾 Sauvegarde de l'index vectoriel...")
        vector_store.save_index()
        
        # Statistiques finales
        stats = vector_store.get_statistics()
        logger.info("📊 Statistiques de l'index:")
        logger.info(f"   Total chunks: {stats['total_chunks']}")
        logger.info(f"   Types de documents: {stats['documents_by_type']}")
        logger.info(f"   Cantons: {len(vector_store.get_available_cantons())}")
        logger.info(f"   Secteurs: {len(vector_store.get_available_sectors())}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération des embeddings: {e}")
        raise

def validate_embeddings():
    """Valide que les embeddings ont été correctement sauvegardés"""
    logger.info("🧪 Validation des embeddings...")
    
    try:
        vector_store = VectorStore(settings)
        
        # Test de chargement
        if vector_store.load_index():
            stats = vector_store.get_statistics()
            logger.info(f"✅ Index chargé avec succès: {stats['total_chunks']} chunks")
            
            # Test de recherche
            test_results = vector_store.search("transport", k=3)
            if test_results:
                logger.info(f"✅ Test de recherche réussi: {len(test_results)} résultats")
                return True
            else:
                logger.error("❌ Test de recherche échoué: aucun résultat")
                return False
        else:
            logger.error("❌ Impossible de charger l'index")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation: {e}")
        return False

def main():
    """Fonction principale optimisée"""
    logger.info("🚀 Génération LÉGÈRE des embeddings pour l'Assistant PECC")
    logger.info("=" * 60)
    
    # Affichage de la configuration
    logger.info("⚙️  Configuration allégée:")
    for key, value in LIGHTWEIGHT_CONFIG.items():
        logger.info(f"   {key}: {value}")
    
    # Monitoring initial
    initial_memory = get_memory_usage()
    logger.info(f"💾 Mémoire initiale: {initial_memory:.1f}MB")
    
    start_total = time.time()
    
    try:
        # 1. Création des dossiers
        logger.info("\nÉtape 1/6: Création des dossiers")
        create_directories()
        
        # 2. Validation des fichiers
        logger.info("\nÉtape 2/6: Validation des fichiers d'entrée")
        if not validate_input_files():
            logger.error("❌ Validation échouée - vérifiez vos fichiers")
            return False
        
        # 3. Traitement léger des documents
        logger.info("\nÉtape 3/6: Traitement léger des documents")
        chunks = process_documents_lightweight()
        
        # 4. Génération légère des embeddings
        logger.info("\nÉtape 4/6: Génération légère des embeddings")
        vector_store = generate_embeddings_lightweight(chunks)
        
        # 5. Validation
        logger.info("\nÉtape 5/6: Validation des embeddings")
        if not validate_embeddings():
            logger.error("❌ Validation des embeddings échouée")
            return False
        
        # 6. Résumé final
        total_time = time.time() - start_total
        final_memory = get_memory_usage()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Génération légère des embeddings terminée avec succès!")
        logger.info(f"⏱️  Temps total: {total_time:.1f} secondes")
        logger.info(f"💾 Mémoire finale: {final_memory:.1f}MB (vs {initial_memory:.1f}MB initial)")
        logger.info(f"📁 Index sauvegardé dans: {settings.VECTOR_DB_PATH}")
        logger.info("✅ L'Assistant PECC est prêt à être utilisé!")
        
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        return False

if __name__ == "__main__":
    # Vérification des ressources système
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    logger.info(f"💾 Mémoire système: {available_memory_gb:.1f}GB disponible / {total_memory_gb:.1f}GB total")
    
    if total_memory_gb < 4:
        logger.warning(f"⚠️  Mémoire système faible: {total_memory_gb:.1f}GB - Mode ultra-léger activé")
        LIGHTWEIGHT_CONFIG['embedding_batch_size'] = 2
        LIGHTWEIGHT_CONFIG['memory_threshold_mb'] = 256
        LIGHTWEIGHT_CONFIG['pause_between_batches'] = 1.0
    elif total_memory_gb < 8:
        logger.info(f"📊 Mémoire modérée: {total_memory_gb:.1f}GB - Mode léger activé")
        LIGHTWEIGHT_CONFIG['embedding_batch_size'] = 4
        LIGHTWEIGHT_CONFIG['memory_threshold_mb'] = 512
    else:
        logger.info(f"✅ Mémoire suffisante: {total_memory_gb:.1f}GB - Mode normal")
        LIGHTWEIGHT_CONFIG['embedding_batch_size'] = 8
        LIGHTWEIGHT_CONFIG['memory_threshold_mb'] = 1024
    
    success = main()
    sys.exit(0 if success else 1)