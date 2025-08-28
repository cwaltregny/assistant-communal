import os
from pathlib import Path
import fitz  

def analyze_pdf_files():
    """Analyse tous les PDFs du projet"""
    print("🔍 Analyse des fichiers PDF")
    print("=" * 50)
    
    folders = {
        "Plans Climat": "data/plan_climat_cantonaux",
        "Mesures": "data/mesures_prioritaires"
    }
    
    all_files = []
    
    for folder_name, folder_path in folders.items():
        print(f"\n📁 {folder_name}: {folder_path}")
        
        if not Path(folder_path).exists():
            print(f"❌ Dossier non trouvé: {folder_path}")
            continue
        
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  Aucun PDF trouvé dans {folder_path}")
            continue
        
        for pdf_file in pdf_files:
            try:
                # Taille du fichier
                size_bytes = pdf_file.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                
                # Informations PDF
                doc = fitz.open(str(pdf_file))
                num_pages = len(doc)
                doc.close()
                
                file_info = {
                    'name': pdf_file.name,
                    'path': str(pdf_file),
                    'size_mb': size_mb,
                    'size_bytes': size_bytes,
                    'pages': num_pages,
                    'folder': folder_name
                }
                
                all_files.append(file_info)
                
                # Couleur selon la taille
                if size_mb > 5:
                    status = "🔴 VOLUMINEUX"
                elif size_mb > 2:
                    status = "🟡 MOYEN"
                else:
                    status = "🟢 OK"
                
                print(f"   {status} {pdf_file.name}")
                print(f"      Taille: {size_mb:.1f}MB ({size_bytes:,} bytes)")
                print(f"      Pages: {num_pages}")
                
            except Exception as e:
                print(f"   ❌ Erreur {pdf_file.name}: {e}")
    
    # Résumé global
    print(f"\n📊 Résumé Global")
    print("=" * 30)
    
    if all_files:
        total_size_mb = sum(f['size_mb'] for f in all_files)
        total_pages = sum(f['pages'] for f in all_files)
        avg_size = total_size_mb / len(all_files)
        
        print(f"📁 Total fichiers: {len(all_files)}")
        print(f"📄 Total pages: {total_pages:,}")
        print(f"💾 Taille totale: {total_size_mb:.1f}MB")
        print(f"📊 Taille moyenne: {avg_size:.1f}MB")
        
        # Fichiers problématiques
        large_files = [f for f in all_files if f['size_mb'] > 5]
        medium_files = [f for f in all_files if 2 < f['size_mb'] <= 5]
        
        if large_files:
            print(f"\n🔴 Fichiers volumineux (>5MB): {len(large_files)}")
            for f in large_files:
                print(f"   {f['name']}: {f['size_mb']:.1f}MB ({f['pages']} pages)")
        
        if medium_files:
            print(f"\n🟡 Fichiers moyens (2-5MB): {len(medium_files)}")
            for f in medium_files:
                print(f"   {f['name']}: {f['size_mb']:.1f}MB ({f['pages']} pages)")
        
        # Recommandations
        print(f"\n💡 Recommandations")
        print("-" * 20)
        
        if large_files:
            print("🔴 Fichiers volumineux détectés:")
            print("   • Risque de crash pendant le traitement")
            print("   • Considérez diviser ces fichiers")
            print("   • Ou augmentez la RAM disponible")
        
        if total_size_mb > 50:
            print("⚠️  Volume total important:")
            print("   • Traitement plus lent")
            print("   • Plus de mémoire requise")
            print("   • Surveillez l'usage RAM")
        
        print(f"\n🔧 Configuration recommandée:")
        if any(f['size_mb'] > 10 for f in all_files):
            print("   • embedding_batch_size: 2")
            print("   • memory_threshold_mb: 256")
            print("   • Pause entre batches: 1.0s")
        elif any(f['size_mb'] > 5 for f in all_files):
            print("   • embedding_batch_size: 4")
            print("   • memory_threshold_mb: 512")
            print("   • Pause entre batches: 0.5s")
        else:
            print("   • Configuration standard OK")
    
    return all_files

def suggest_optimizations():
    """Suggestions d'optimisation"""
    print(f"\n🎯 Optimisations Possibles")
    print("=" * 30)
    
    print("1. 📄 Réduire la taille des PDFs:")
    print("   • Compresser les images dans les PDFs")
    print("   • Exporter en qualité réduite")
    print("   • Diviser les gros documents")
    
    print("\n2. ⚙️ Ajuster la configuration:")
    print("   • Réduire CHUNK_SIZE dans config.py")
    print("   • Diminuer embedding_batch_size")
    print("   • Augmenter pause_between_batches")
    
    print("\n3. 💾 Améliorer le système:")
    print("   • Fermer autres applications")
    print("   • Augmenter la RAM si possible")
    print("   • Utiliser un SSD")

def main():
    """Fonction principale"""
    files = analyze_pdf_files()
    
    if files:
        suggest_optimizations()
        
        # Test rapide de mémoire
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"\n💾 Mémoire système actuelle:")
            print(f"   Total: {memory.total / (1024**3):.1f}GB")
            print(f"   Disponible: {memory.available / (1024**3):.1f}GB")
            print(f"   Utilisée: {memory.percent:.1f}%")
        except ImportError:
            print("\n💾 Installez psutil pour le monitoring mémoire:")
            print("   pip install psutil")
    
    return True

if __name__ == "__main__":
    main()