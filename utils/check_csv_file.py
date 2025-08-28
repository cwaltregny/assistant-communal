import pandas as pd
import sys
from pathlib import Path

def analyze_csv(csv_path="data/donnees_communes_clean.csv"):
    """Analyse détaillée du CSV"""
    try:
        print(f"🔍 Analyse du fichier: {csv_path}")
        print("=" * 50)
        
        # Lecture du CSV
        df = pd.read_csv(csv_path)
        
        print(f"📊 Dimensions: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        print(f"📋 Colonnes: {list(df.columns)}")
        
        print("\n🔍 Types de données détectés:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            print(f"   {col}: {dtype} ({null_count} nulls, {unique_count} uniques)")
        
        print("\n📋 Échantillon des premières lignes:")
        print(df.head(3).to_string())
        
        print("\n🔍 Analyse par colonne:")
        
        # Analyse de la colonne commune
        print(f"\n📍 Colonne 'commune':")
        if 'commune' in df.columns:
            communes = df['commune']
            print(f"   Type: {communes.dtype}")
            print(f"   Exemples: {list(communes.head(5))}")
            print(f"   Valeurs nulles: {communes.isnull().sum()}")
        
        # Analyse de la colonne population
        print(f"\n👥 Colonne 'population':")
        if 'population' in df.columns:
            population = df['population']
            print(f"   Type: {population.dtype}")
            print(f"   Exemples: {list(population.head(5))}")
            print(f"   Valeurs nulles: {population.isnull().sum()}")
            
            # Test de conversion
            try:
                if population.dtype == 'object':
                    # Essai de conversion string vers int
                    test_values = population.dropna().head(5)
                    print(f"   Test conversion: {[str(v).strip() for v in test_values]}")
                else:
                    print(f"   Déjà numérique")
            except Exception as e:
                print(f"   Erreur test conversion: {e}")
        
        # Analyse de la colonne canton
        print(f"\n🏛️ Colonne 'canton':")
        if 'canton' in df.columns:
            cantons = df['canton'].value_counts()
            print(f"   Cantons uniques: {list(cantons.index)}")
            print(f"   Répartition: {dict(cantons)}")
        
        # Analyse de la colonne pecc
        print(f"\n📋 Colonne 'pecc':")
        if 'pecc' in df.columns:
            pecc_values = df['pecc'].value_counts()
            print(f"   Valeurs PECC: {dict(pecc_values)}")
        
        # Analyse des valeurs problématiques
        print(f"\n⚠️  Détection de problèmes potentiels:")
        
        for col in df.columns:
            col_data = df[col]
            
            # Valeurs vides ou nulles
            null_count = col_data.isnull().sum()
            if null_count > 0:
                print(f"   {col}: {null_count} valeurs nulles")
            
            # Types mixtes (pour colonnes supposées numériques)
            if col in ['population', 'superficie', 'densite', 'ofs']:
                if col_data.dtype == 'object':
                    print(f"   {col}: Type 'object' au lieu de numérique")
                    
                    # Échantillon de valeurs non-numériques
                    try:
                        sample_values = col_data.dropna().head(10)
                        non_numeric = []
                        for val in sample_values:
                            try:
                                float(str(val).replace(',', '.').replace(' ', ''))
                            except:
                                non_numeric.append(val)
                        
                        if non_numeric:
                            print(f"      Valeurs non-numériques: {non_numeric[:5]}")
                    except Exception as e:
                        print(f"      Erreur analyse: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        print(f"Détails: {traceback.format_exc()}")
        return False

def test_municipality_loading():
    """Test la fonction de chargement des communes"""
    print(f"\n🧪 Test de chargement des communes")
    print("=" * 50)
    
    try:
        # Import local
        sys.path.append('.')
        from process_documents import DocumentProcessor
        from config_ac import settings
        
        processor = DocumentProcessor(settings)
        municipalities = processor.load_municipalities_data()
        
        if municipalities:
            print(f"✅ Chargement réussi: {len(municipalities)} communes")
            
            # Affichage d'exemples
            print(f"\n📋 Exemples de communes chargées:")
            for i, (key, data) in enumerate(list(municipalities.items())[:3]):
                print(f"   {i+1}. {data['nom']} ({data['canton']})")
                print(f"      Population: {data['population']:,}")
                print(f"      Superficie: {data['superficie']} km²")
                print(f"      PECC: {data['pecc']}")
                print()
            
            # Statistiques
            total_pop = sum(m['population'] for m in municipalities.values())
            avec_pecc = sum(1 for m in municipalities.values() if m['has_pecc'])
            
            print(f"📊 Statistiques:")
            print(f"   Population totale: {total_pop:,} habitants")
            print(f"   Communes avec PECC: {avec_pecc}/{len(municipalities)}")
            
            return True
        else:
            print("❌ Aucune commune chargée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        print(f"Détails: {traceback.format_exc()}")
        return False

def main():
    """Fonction principale de debug"""
    print("🔧 Debug CSV Assistant PECC")
    print("=" * 50)
    
    csv_path = "data/donnees_communes_clean.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        print("   Vérifiez le chemin du fichier")
        return False
    
    # 1. Analyse du CSV brut
    if not analyze_csv(csv_path):
        return False
    
    # 2. Test de chargement
    if not test_municipality_loading():
        return False
    
    print("\n🎉 Debug terminé avec succès!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)