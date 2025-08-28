import pandas as pd
import sys
from pathlib import Path
import re

NBSP = u"\xa0"

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Normalize Swiss-formatted numbers to a standard '1234.56' string:
    - remove non-breaking spaces & regular spaces used as thousands separators
    - remove apostrophes used as thousands separators
    - replace decimal comma with dot
    - keep only the first valid numeric pattern (handles stray text)
    """
    if s.dtype != "object":
        s = s.astype(str)

    s = (
        s.astype(str)
         .str.replace(NBSP, "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("’", "", regex=False)   # curly apostrophe
         .str.replace("'", "", regex=False)   # straight apostrophe
         .str.replace("\u202f", "", regex=False)  # narrow NBSP if present
         .str.replace(",", ".", regex=False)
    )
    # Extract first numeric like -1234.56 from the string (drops stray text)
    extracted = s.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return extracted

def _to_int(series: pd.Series, round_before=True) -> pd.Series:
    cleaned = _clean_numeric_series(series)
    num = pd.to_numeric(cleaned, errors="coerce")
    if round_before:
        num = num.round(0)
    # Use pandas nullable integer to allow NaN without crashing
    return num.astype("Int64")

def _to_float(series: pd.Series) -> pd.Series:
    cleaned = _clean_numeric_series(series)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)

def analyze_csv(csv_path="data/donnees_communes.csv"):
    """Analyse détaillée du CSV + conversions typées"""
    try:
        print(f"🔍 Analyse du fichier: {csv_path}")
        print("=" * 50)

        # 1) Lire tout en texte pour éviter les inférences bancales
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])

        print(f"📊 Dimensions: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        print(f"📋 Colonnes: {list(df.columns)}")

        print("\n🔍 Types initiaux:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")

        # 2) Conversions demandées
        if "population" in df.columns:
            df["population"] = _to_int(df["population"], round_before=False)  # population should be exact integer
        if "superficie" in df.columns:
            df["superficie"] = _to_float(df["superficie"])
        if "densite" in df.columns:
            df["densite"] = _to_float(df["densite"])

        # Optionally: make 'ofs' integer as well if present
        if "ofs" in df.columns:
            df["ofs"] = _to_int(df["ofs"], round_before=False)

        print("\n✅ Types après conversion:")
        for col in ["population", "superficie", "densite", "ofs"]:
            if col in df.columns:
                print(f"   {col}: {df[col].dtype}")

        # 3) Aperçu
        print("\n📋 Échantillon (après conversion):")
        display_cols = [c for c in ["commune", "ofs", "population", "superficie", "densite", "canton", "pecc"] if c in df.columns]
        print(df[display_cols].head(5).to_string(index=False))

        # 4) Contrôles rapides
        for col, kind in [("population", "int"), ("superficie", "int"), ("densite", "float")]:
            if col in df.columns:
                na = df[col].isna().sum()
                print(f"   {col}: {na} valeurs non convertibles (NaN)")
                
        df.to_csv('data/donnees_communes_clean.csv')

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

            print(f"\n📋 Exemples de communes chargées:")
            for i, (key, data) in enumerate(list(municipalities.items())[:3]):
                print(f"   {i+1}. {data['nom']} ({data['canton']})")
                print(f"      Population: {data['population']:,}")
                print(f"      Superficie: {data['superficie']} km²")
                print(f"      PECC: {data['pecc']}")
                print()

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

    csv_path = "data/donnees_communes.csv"

    if not Path(csv_path).exists():
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        print("   Vérifiez le chemin du fichier")
        return False

    if not analyze_csv(csv_path):
        return False

    if not test_municipality_loading():
        return False

    print("\n🎉 Debug terminé avec succès!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)