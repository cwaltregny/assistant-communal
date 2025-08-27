import os
import csv
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import fitz  
from pathlib import Path
from dataclasses import dataclass
import re

@dataclass
class DocumentChunk:
    content: str
    source_file: str
    page_number: int
    chunk_id: str
    document_type: str  
    metadata: Dict

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.municipalities_data = None
        self.logger = logging.getLogger(__name__)
        
    def load_municipalities_data(self) -> Dict:
        """Charge les données des communes depuis le CSV"""
        try:
            df = pd.read_csv(self.config.MUNICIPALITIES_CSV)
            # Normalise les noms de colonnes
            df.columns = df.columns.str.lower().str.strip()
            
            municipalities = {}
            for _, row in df.iterrows():
                # Gestion sécurisée du nom de commune
                commune_name = str(row.get('commune', '')).strip() if pd.notna(row.get('commune')) else ''
                
                if commune_name:
                    # Conversion sécurisée des valeurs numériques
                    def safe_convert_to_int(value, default=0):
                        try:
                            if pd.isna(value):
                                return default
                            # Si c'est déjà un nombre
                            if isinstance(value, (int, float)):
                                return int(value)
                            # Si c'est une string, nettoie et convertit
                            cleaned = str(value).strip().replace(' ', '').replace(',', '')
                            return int(float(cleaned)) if cleaned else default
                        except (ValueError, TypeError):
                            return default
                    
                    def safe_convert_to_float(value, default=0.0):
                        try:
                            if pd.isna(value):
                                return default
                            # Si c'est déjà un nombre
                            if isinstance(value, (int, float)):
                                return float(value)
                            # Si c'est une string, nettoie et convertit
                            cleaned = str(value).strip().replace(' ', '').replace(',', '.')
                            return float(cleaned) if cleaned else default
                        except (ValueError, TypeError):
                            return default
                    
                    def safe_string(value, default=''):
                        try:
                            if pd.isna(value):
                                return default
                            return str(value).strip()
                        except:
                            return default
                    
                    # Extraction sécurisée de toutes les données
                    population = safe_convert_to_int(row.get('population'), 0)
                    superficie = safe_convert_to_float(row.get('superficie'), 0.0)
                    densite = safe_convert_to_float(row.get('densite'), 0.0)
                    canton = safe_string(row.get('canton'), '')
                    pecc_status = safe_string(row.get('pecc'), '')
                    details = safe_string(row.get('details'), '')
                    ofs = safe_convert_to_int(row.get('ofs'), 0)
                    
                    municipalities[commune_name.lower()] = {
                        'nom': commune_name,
                        'ofs': ofs,
                        'population': population,
                        'superficie': superficie,
                        'densite': densite,
                        'canton': canton,
                        'pecc': pecc_status,
                        'has_pecc': pecc_status.lower() in ['oui', 'yes', 'true', '1'],
                        'details': details,
                        'habitants': population,
                        'altitude': 0  
                    }
            
            self.municipalities_data = municipalities
            self.logger.info(f"Chargé {len(municipalities)} communes")
            
            total_population = sum(m['population'] for m in municipalities.values())
            communes_avec_pecc = sum(1 for m in municipalities.values() if m['has_pecc'])
            
            self.logger.info(f"   Population totale: {total_population:,} habitants")
            self.logger.info(f"   Communes avec PECC: {communes_avec_pecc}/{len(municipalities)}")
            
            return municipalities
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données des communes: {e}")
            # Debug détaillé
            if hasattr(e, '__traceback__'):
                import traceback
                self.logger.error(f"Détails: {traceback.format_exc()}")
            return {}

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extrait le texte d'un PDF avec numéro de page - Version optimisée mémoire"""
        try:
            file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
            self.logger.info(f"Traitement PDF: {Path(pdf_path).name} ({file_size_mb:.1f}MB)")
            
            doc = fitz.open(pdf_path)
            pages_text = []
            total_pages = len(doc)
            
            # Traitement page par page pour économiser la mémoire
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Nettoyage du texte
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    
                    if text and len(text) > 50:  # Ignore les pages quasi-vides
                        pages_text.append((text, page_num + 1))
                    
                    # Nettoyage de la page pour libérer la mémoire
                    page = None
                    
                    # Progress pour gros documents
                    if total_pages > 50 and page_num % 20 == 0:
                        self.logger.info(f"   Page {page_num + 1}/{total_pages}")
                        
                    # Garbage collection périodique pour gros documents
                    if file_size_mb > 1 and page_num % 10 == 0:
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    self.logger.warning(f"Erreur page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            doc = None  # Libère explicitement
            
            self.logger.info(f"   ✅ {len(pages_text)} pages extraites")
            return pages_text
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'extraction de {pdf_path}: {e}")
            return []

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Divise le texte en chunks avec overlap - Version optimisée"""
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        # Pour de très longs textes, réduit la taille des chunks
        if len(text) > 50000:
            chunk_size = min(chunk_size, 800)
            overlap = min(overlap, 100)
            self.logger.info(f"   Texte long détecté, chunks réduits: {chunk_size}")
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Essaie de couper à la fin d'une phrase
            if end < len(text):
                # Cherche dans les 200 derniers caractères
                search_zone = text[max(start, end-200):end]
                last_period = search_zone.rfind('.')
                last_newline = search_zone.rfind('\n')
                
                if last_period > 0:
                    end = start + max(0, end - 200) + last_period + 1
                elif last_newline > 0:
                    end = start + max(0, end - 200) + last_newline + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 20:  # Ignore les chunks trop petits
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
            
            # Sécurité anti-boucle infinie
            if start >= len(text):
                break
        
        return chunks

    def process_plan_climat_documents(self) -> List[DocumentChunk]:
        """Traite tous les documents Plan Climat Cantonaux"""
        chunks = []
        plan_climat_folder = Path(self.config.PLAN_CLIMAT_FOLDER)
        
        if not plan_climat_folder.exists():
            self.logger.warning(f"Dossier Plan Climat non trouvé: {plan_climat_folder}")
            return chunks
        
        for pdf_file in plan_climat_folder.glob("*.pdf"):
            self.logger.info(f"Traitement du Plan Climat: {pdf_file.name}")
            
            # Extrait le canton du nom de fichier
            canton = self._extract_canton_from_filename(pdf_file.name)
            
            pages_text = self.extract_text_from_pdf(str(pdf_file))
            
            for text, page_num in pages_text:
                text_chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(text_chunks):
                    chunk_obj = DocumentChunk(
                        content=chunk,
                        source_file=pdf_file.name,
                        page_number=page_num,
                        chunk_id=f"{pdf_file.stem}_p{page_num}_c{i}",
                        document_type="plan_climat",
                        metadata={
                            "canton": canton,
                            "file_path": str(pdf_file),
                            "total_chunks_page": len(text_chunks),
                            "chunk_index": i
                        }
                    )
                    chunks.append(chunk_obj)
        
        self.logger.info(f"Créé {len(chunks)} chunks depuis les Plans Climat")
        return chunks

    def process_priority_measures_documents(self) -> List[DocumentChunk]:
        """Traite tous les documents de mesures prioritaires"""
        chunks = []
        measures_folder = Path(self.config.PRIORITY_MEASURES_FOLDER)
        
        if not measures_folder.exists():
            self.logger.warning(f"Dossier mesures prioritaires non trouvé: {measures_folder}")
            return chunks
        
        for pdf_file in measures_folder.glob("*.pdf"):
            self.logger.info(f"Traitement de la mesure: {pdf_file.name}")
            
            # Extrait le secteur du nom de fichier
            secteur = self._extract_sector_from_filename(pdf_file.name)
            
            pages_text = self.extract_text_from_pdf(str(pdf_file))
            
            for text, page_num in pages_text:
                text_chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(text_chunks):
                    chunk_obj = DocumentChunk(
                        content=chunk,
                        source_file=pdf_file.name,
                        page_number=page_num,
                        chunk_id=f"{pdf_file.stem}_p{page_num}_c{i}",
                        document_type="mesure_prioritaire",
                        metadata={
                            "secteur": secteur,
                            "file_path": str(pdf_file),
                            "total_chunks_page": len(text_chunks),
                            "chunk_index": i
                        }
                    )
                    chunks.append(chunk_obj)
        
        self.logger.info(f"Créé {len(chunks)} chunks depuis les mesures prioritaires")
        return chunks

    def _extract_canton_from_filename(self, filename: str) -> str:
        """Extrait le canton du nom de fichier du Plan Climat"""
        filename_lower = filename.lower()
        
        # Mapping spécifique pour vos fichiers
        canton_mapping = {
            'pcc_fribourg': 'Fribourg',
            'pcc_geneve': 'Genève', 
            'pcc_neuchatel': 'Neuchâtel',
            'pcc_valais': 'Valais',
            'pcc_vaud': 'Vaud'
        }
        
        # Extrait le nom de base du fichier (sans extension)
        base_name = filename_lower.replace('.pdf', '')
        
        if base_name in canton_mapping:
            return canton_mapping[base_name]
        
        # Fallback: recherche de mots-clés
        for key, canton in canton_mapping.items():
            if key.split('_')[1] in filename_lower:  # Ex: 'fribourg' dans le nom
                return canton
        
        return "Canton_Inconnu"

    def _extract_sector_from_filename(self, filename: str) -> str:
        """Extrait le secteur du nom de fichier de mesure"""
        filename_lower = filename.lower()
        
        # Mapping spécifique pour vos fichiers de mesures
        sector_mapping = {
            'mobilité': 'Transport',
            'transport': 'Transport',
            'alimentation': 'Alimentation',
            'amenagement': 'Aménagement du territoire',
            'territoire': 'Aménagement du territoire',
            'construction': 'Bâtiment',
            'bois': 'Bâtiment',
            'energie': 'Energie',
            'renovation': 'Bâtiment',
            'energetique': 'Energie',
            'batiment': 'Bâtiment',
            'stc': 'Aménagement du territoire'  
        }
        
        # Recherche des mots-clés dans le nom de fichier
        for keyword, secteur in sector_mapping.items():
            if keyword in filename_lower:
                return secteur
        
        # Cas spéciaux pour vos fichiers
        if 'bons_et_moins_bons_exemples' in filename_lower:
            return 'Transport'
        elif 'modes_transport_alternatifs' in filename_lower:
            return 'Transport'
        elif 'stc-note' in filename_lower:
            return 'Aménagement du territoire'
        
        return "Secteur_Inconnu"

    def get_municipality_info(self, municipality_name: str) -> Optional[Dict]:
        """Récupère les informations d'une commune"""
        if not self.municipalities_data:
            self.load_municipalities_data()
        
        # Recherche exact
        municipality_lower = municipality_name.lower().strip()
        if municipality_lower in self.municipalities_data:
            return self.municipalities_data[municipality_lower]
        
        # Recherche approximative
        for name, info in self.municipalities_data.items():
            if municipality_lower in name or name in municipality_lower:
                return info
        
        return None

    def process_all_documents(self) -> List[DocumentChunk]:
        """Traite tous les documents"""
        all_chunks = []
        
        # Charge les données des communes
        self.load_municipalities_data()
        
        # Traite les Plans Climat
        plan_climat_chunks = self.process_plan_climat_documents()
        all_chunks.extend(plan_climat_chunks)
        
        # Traite les mesures prioritaires
        priority_chunks = self.process_priority_measures_documents()
        all_chunks.extend(priority_chunks)
        
        self.logger.info(f"Total: {len(all_chunks)} chunks traités")
        return all_chunks