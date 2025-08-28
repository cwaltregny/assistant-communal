import logging
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime
from pathlib import Path
from process_documents import DocumentProcessor
from client_mistral import MistralClient

class SimplifiedConversationManager:
    """Gestionnaire de conversation simplifié"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        """Remet à zéro la conversation"""
        self.messages = []
        self.context = {
            "current_municipality": None,
            "current_sector": None,
            "current_canton": None,
            "user_preferences": {}
        }
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Ajoute un message à l'historique"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Garde seulement les derniers messages
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_conversation_context(self) -> str:
        """Génère un contexte de conversation simple"""
        if not self.messages:
            return ""
        
        context_parts = ["Contexte de la conversation précédente:"]
        
        # Ajoute les derniers échanges
        recent_messages = self.messages[-4:]  # 2 derniers échanges
        for msg in recent_messages:
            role_fr = "Utilisateur" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_fr}: {msg['content'][:150]}...")
        
        # Ajoute le contexte persistant
        if self.context["current_municipality"]:
            context_parts.append(f"Commune en cours de discussion: {self.context['current_municipality']}")
        
        if self.context["current_sector"]:
            context_parts.append(f"Secteur en cours de discussion: {self.context['current_sector']}")
        
        return "\n".join(context_parts)
    
    def update_context(self, analysis: Dict):
        """Met à jour le contexte de conversation"""
        if analysis.get("municipality"):
            self.context["current_municipality"] = analysis["municipality"]
        
        if analysis.get("sector"):
            self.context["current_sector"] = analysis["sector"]
        
        if analysis.get("canton"):
            self.context["current_canton"] = analysis["canton"]
    
    def get_context_summary(self) -> Dict:
        """Retourne un résumé du contexte actuel"""
        return {
            "message_count": len(self.messages),
            "current_municipality": self.context.get("current_municipality"),
            "current_sector": self.context.get("current_sector"),
            "current_canton": self.context.get("current_canton"),
            "last_interaction": self.messages[-1]["timestamp"] if self.messages else None
        }

class PECCAssistant:
    """Assistant PECC avec recherche hybride"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.vector_store = None
        
        self.document_processor = DocumentProcessor(config)
        self.mistral_client = MistralClient(config)
        
        self.conversation_manager = SimplifiedConversationManager()
        
        self.system_prompt = self._get_system_prompt()
        
        self.secteurs_disponibles = [
            "Transport", "Bâtiment", "Energie", "Alimentation", "Aménagement du territoire"
        ]
    
    def _get_system_prompt(self):
        """Génère le prompt système optimisé"""
        
        return """Tu es un expert en politique climatique en Suisse romande et un assistant conversationnel pour les communes suisses. Tu aides à élaborer des PECC (Plans Energie Climat Communaux) et à identifier les mesures clés pour la décarbonation et l'adaptation.

INSTRUCTIONS IMPORTANTES:
- Tu réponds toujours en français, de façon claire et structurée
- Tu maintiens une conversation fluide en te référant aux échanges précédents
- Si une commune ou secteur a été mentionné, tu t'en souviens
- Tu peux faire des références contextuelles
- Si la question concerne une commune précise, commence par regarder le nombre d'habitants
- Pour les questions générales sur les mesures, demande de préciser le secteur si pas mentionné
- Si tu ne trouves pas la réponse dans les documents, dis-le clairement

SECTEURS DISPONIBLES: Transport, Bâtiment, Energie, Alimentation, Aménagement du territoire

CAPACITÉS AVANCÉES:
- Tu utilises la recherche hybride pour des résultats plus pertinents
- Tu bénéficies du reranking pour améliorer l'ordre des sources
- Tu priorises les informations les plus fiables grâce aux scores avancés
- Tu synthétises les informations de multiple sources de façon optimale"""
    
    async def initialize(self):
        """Initialise l'assistant avec le vector store avancé"""
        self.logger.info("Initialisation de l'assistant PECC...")
        
        try:
            from enhanced_vector_store import EnhancedVectorStore
            
            enhanced_settings = {
                'index_path': 'vector_store_enhanced',
                'hybrid_search_enabled': True,
                'rerank_enabled': True,
                'query_expansion_enabled': True,
                'semantic_weight': 0.7,
                'lexical_weight': 0.3,
                'rerank_top_k': 20,
                'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            }
            
            if hasattr(self.config, '__dict__'):
                for key, value in self.config.__dict__.items():
                    if not key.startswith('_') and key not in enhanced_settings:
                        enhanced_settings[key] = value
            
            self.vector_store = EnhancedVectorStore(enhanced_settings)
            
            if not self.vector_store.load_index():
                raise ValueError("Impossible de charger l'index du vector store avancé")
                
        except ImportError:
            self.logger.error("❌ enhanced_vector_store.py non trouvé")
            self.logger.error("   Assurez-vous d'avoir créé le fichier avec le code fourni")
            raise
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement du vector store: {e}")
            raise
        
        # Charge les données des communes
        self.document_processor.load_municipalities_data()
        
        # Log des capacités
        stats = self.vector_store.get_statistics()
        self.logger.info(f"Assistant initialisé avec {stats.get('total_chunks', 0)} chunks")
        self.logger.info("🚀 Capacités avancées:")
        self.logger.info(f"   - Recherche hybride: {stats.get('hybrid_search', 'N/A')}")
        self.logger.info(f"   - Reranking: {stats.get('reranking_enabled', 'N/A')}")
        self.logger.info(f"   - Features TF-IDF: {stats.get('lexical_features', 'N/A')}")
        
        self.logger.info("Assistant PECC initialisé avec succès")
    
    def reset_conversation(self):
        """Démarre une nouvelle conversation"""
        self.conversation_manager.reset()
        self.logger.info("Nouvelle conversation initialisée")
    
    async def process_message(self, user_message: str, session_id: str = None) -> Dict:
        """Traite un message utilisateur"""
        try:
            # Ajoute le message utilisateur à l'historique
            self.conversation_manager.add_message("user", user_message)
            
            # Analyse la requête avec contexte
            query_analysis = self._analyze_query_with_context(user_message)
            
            # Met à jour le contexte de conversation
            self.conversation_manager.update_context(query_analysis)
            
            # Récupération de documents intelligente
            relevant_docs = await self._intelligent_document_retrieval(user_message, query_analysis)
            
            # Génère la réponse
            response = await self._generate_response(user_message, relevant_docs, query_analysis)
            
            # Ajoute la réponse à l'historique
            response_metadata = {
                "query_type": query_analysis["type"],
                "municipality": query_analysis.get("municipality"),
                "sector": query_analysis.get("sector"),
                "search_method": query_analysis.get("search_method", "hybrid"),
                "sources_count": len(relevant_docs)
            }
            
            self.conversation_manager.add_message("assistant", response, response_metadata)
            
            # Formate les références
            references = self._format_references(relevant_docs)
            
            result = {
                "response": response,
                "references": references,
                "query_type": query_analysis["type"],
                "municipality": query_analysis.get("municipality"),
                "sector": query_analysis.get("sector"),
                "conversation_context": self.conversation_manager.get_context_summary(),
                "search_method": query_analysis.get("search_method"),
                "search_quality": self._assess_search_quality(relevant_docs),
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du message: {e}")
            error_response = "Désolé, une erreur s'est produite lors du traitement de votre message."
            
            self.conversation_manager.add_message("assistant", error_response)
            
            return {
                "response": error_response,
                "references": [],
                "status": "error",
                "error": str(e)
            }
    
    async def _intelligent_document_retrieval(self, query: str, analysis: Dict) -> List[Dict]:
        """Récupération intelligente de documents avec le vector store avancé"""
        try:
            # Méthode de recherche selon l'analyse
            search_method = analysis.get("search_method", "hybrid")
            
            # Recherche spécialisée selon le type de requête
            if analysis["type"] == "municipality_specific" and analysis.get("canton"):
                # Recherche hybride dans les documents du canton
                canton_docs = self.vector_store.search_by_canton(query, analysis["canton"], k=3)
                
                # Recherche générale complémentaire
                general_docs = self.vector_store.search(query, k=5, method=search_method)
                
                # Combine et déduplique
                all_docs = canton_docs + general_docs
                return self._deduplicate_results(all_docs)
                
            elif analysis["type"] == "sector_specific" and analysis.get("sector"):
                # Recherche spécialisée par secteur
                sector_docs = self.vector_store.search_by_sector(query, analysis["sector"], k=5)
                
                if len(sector_docs) < 3:
                    complementary_docs = self.vector_store.search(query, k=5, method=search_method)
                    sector_docs.extend(complementary_docs)
                
                return self._deduplicate_results(sector_docs)
            else:
                return self.vector_store.search(query, k=5, method=search_method)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération intelligente: {e}")
            return []
    
    def _deduplicate_results(self, docs: List[Dict]) -> List[Dict]:
        """Supprime les doublons en gardant les meilleurs scores"""
        seen = {}
        for doc in docs:
            chunk_id = doc.get("chunk_id", f"{doc.get('source_file', '')}_{doc.get('page_number', 0)}")
            if chunk_id not in seen or doc.get("similarity_score", 0) > seen[chunk_id].get("similarity_score", 0):
                seen[chunk_id] = doc
        
        return sorted(seen.values(), key=lambda x: x.get("similarity_score", 0), reverse=True)
    
    def _analyze_query_with_context(self, query: str) -> Dict:
        """Analyse avancée de la requête avec contexte conversationnel"""
        # Analyse de base
        analysis = self._analyze_query(query)
        
        # Ajoute des informations contextuelles
        current_context = self.conversation_manager.context
        
        contextual_indicators = ["cette", "ce", "la commune", "le secteur", "là-bas", "ici"]
        has_contextual_reference = any(indicator in query.lower() for indicator in contextual_indicators)
        
        if has_contextual_reference:
            analysis["context_aware"] = True
            if current_context.get("current_municipality") and not self._has_explicit_municipality(query):
                analysis["municipality"] = current_context["current_municipality"]
                analysis["type"] = "municipality_specific"
            
            if current_context.get("current_sector") and not self._has_explicit_sector(query):
                analysis["sector"] = current_context["current_sector"]
                if analysis["type"] == "general":
                    analysis["type"] = "sector_specific"
        
        # Définit la méthode de recherche (toujours hybride pour ce store)
        analysis["search_method"] = "hybrid"
        
        return analysis
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyse standard de la requête"""
        query_lower = query.lower()
        
        analysis = {
            "type": "general",
            "municipality": None,
            "sector": None,
            "canton": None
        }
        
        # Détecte les communes
        municipality_patterns = [
        r'\b(?:commune|ville|municipalité)\s+(?:de\s+)?([A-Za-zÀ-ÿ\- ]+?)\s+(?=et\b|qui\b|je\b|j\'|nous\b|,|\.|$)',
        r'\b([A-Za-zÀ-ÿ\-]+)(?=\s+(?:commune|ville|municipalité))',
        r'\bà\s+([A-Za-zÀ-ÿ\- ]+?)\s+(?=et\b|qui\b|je\b|j\'|nous\b|,|\.|$)',
        r'\bpour\s+([A-Za-zÀ-ÿ\- ]+?)\s+(?=et\b|qui\b|je\b|j\'|nous\b|,|\.|$)'
        ]
        
        for pattern in municipality_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_municipality = match.group(1).strip()
                municipality_info = self.document_processor.get_municipality_info(potential_municipality)
                if municipality_info:
                    analysis["municipality"] = potential_municipality
                    analysis["canton"] = municipality_info.get("canton")
                    analysis["type"] = "municipality_specific"
                    break
        
        # Détecte les secteurs
        for secteur in self.secteurs_disponibles:
            if secteur.lower() in query_lower:
                analysis["sector"] = secteur
                if analysis["type"] == "general":
                    analysis["type"] = "sector_specific"
                break
        
        # Détecte les demandes de mesures
        if any(word in query_lower for word in ["mesure", "action", "recommandation", "priorité"]):
            if analysis["type"] == "general":
                analysis["type"] = "measures_request"
        
        return analysis
    
    def _has_explicit_municipality(self, query: str) -> bool:
        """Vérifie si la requête mentionne explicitement une commune"""
        municipality_patterns = [
            r'\b(?:commune|ville|municipalité)\s+(?:de\s+)?([A-Za-z\-\s]+)',
            r'\b([A-Za-z\-]+)\s+(?:commune|ville|municipalité)',
            r'\bà\s+([A-Za-z\-\s]+)\b',
            r'\bpour\s+([A-Za-z\-\s]+)\b'
        ]
        
        for pattern in municipality_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _has_explicit_sector(self, query: str) -> bool:
        """Vérifie si la requête mentionne explicitement un secteur"""
        query_lower = query.lower()
        return any(secteur.lower() in query_lower for secteur in self.secteurs_disponibles)
    
    async def _generate_response(self, query: str, relevant_docs: List[Dict], analysis: Dict) -> str:
        """Génère la réponse avec le contexte approprié"""
        
        # Construit le contexte de conversation
        conversation_context = self.conversation_manager.get_conversation_context()
        
        # Enrichit la requête
        enhanced_query = query
        
        # Ajoute le contexte de conversation si pertinent
        if conversation_context and analysis.get("context_aware"):
            enhanced_query = f"{conversation_context}\n\nNouvelle question: {query}"
        
        # Ajoute les informations de commune si applicable
        if analysis.get("municipality"):
            municipality_info = self.document_processor.get_municipality_info(analysis["municipality"])
            if municipality_info:
                context_info = f"\nInformations sur {analysis['municipality']}: "
                context_info += f"Canton: {municipality_info.get('canton', 'N/A')}, "
                context_info += f"Population: {municipality_info.get('population', 'N/A')} habitants, "
                context_info += f"Superficie: {municipality_info.get('superficie', 'N/A')} km², "
                context_info += f"PECC existant: {municipality_info.get('pecc', 'Non renseigné')}"
                
                # Ajoute les détails si disponibles
                details = municipality_info.get('details', '')
                if details and len(details) > 50:
                    context_info += f"\nDétails: {details[:500]}..."
                
                enhanced_query = context_info + "\n\nQuestion: " + enhanced_query
        
        # Gère les demandes de clarification
        if analysis["type"] == "measures_request" and not analysis.get("sector") and not analysis.get("municipality"):
            sectors_list = ", ".join(self.secteurs_disponibles)
            return f"""L'utilisateur demande des informations sur les mesures prioritaires mais n'a pas spécifié de secteur. 

Demande-lui de préciser sur quel secteur il souhaite se concentrer parmi: {sectors_list}.

Question originale: {query}"""
        
        # Génère la réponse
        return await self.mistral_client.generate_response(
            self.system_prompt,
            enhanced_query,
            relevant_docs
        )
    
    def _assess_search_quality(self, documents: List[Dict]) -> Dict:
        """Évalue la qualité des résultats de recherche"""
        if not documents:
            return {"quality": "poor", "reason": "no_results"}
        
        # Calcule des métriques de qualité
        scores = [doc.get("similarity_score", 0) for doc in documents]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Vérifie la diversité des sources
        sources = set(doc.get("source_file", "") for doc in documents)
        source_diversity = len(sources) / len(documents) if documents else 0
        
        # Vérifie la présence de scores avancés
        has_advanced_scores = any(
            doc.get("semantic_score", 0) > 0 or doc.get("lexical_score", 0) > 0 
            for doc in documents
        )
        
        # Détermine la qualité globale
        if max_score > 0.8 and avg_score > 0.6:
            quality = "excellent"
        elif max_score > 0.6 and avg_score > 0.4:
            quality = "good"
        elif max_score > 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "avg_score": round(avg_score, 3),
            "max_score": round(max_score, 3),
            "source_diversity": round(source_diversity, 3),
            "has_advanced_scores": has_advanced_scores,
            "num_results": len(documents)
        }
    
    def _format_references(self, documents: List[Dict]) -> List[Dict]:
        """Formate les références avec informations complètes"""
        references = []
        
        for doc in documents:
            content = doc.get("content", doc.get("extract", ""))
            
            # Limite la taille de l'extrait pour les performances
            if len(content) > 500:
                cut_point = content.find(' ', 350)
                if cut_point == -1 or cut_point > 500:
                    cut_point = 400
                extract = content[:cut_point] + "..."
            else:
                extract = content
            
            ref = {
                "source_file": doc.get("source_file", ""),
                "page_number": doc.get("page_number", 0),
                "document_type": doc.get("document_type", ""),
                "similarity_score": round(doc.get("similarity_score", 0), 3),
                "extract": extract
            }
            
            # Ajoute les scores avancés si disponibles
            for score_type in ["semantic_score", "lexical_score", "rerank_score"]:
                if score_type in doc:
                    ref[score_type] = round(doc[score_type], 3)
            
            if "search_method" in doc:
                ref["search_method"] = doc["search_method"]
            
            # Métadonnées spécifiques au domaine
            if doc.get("document_type") == "plan_climat":
                ref["canton"] = doc.get("metadata", {}).get("canton")
            elif doc.get("document_type") == "mesure_prioritaire":
                ref["secteur"] = doc.get("metadata", {}).get("secteur")
            
            references.append(ref)
        
        return references
    
    async def stream_conversation_with_metadata(self, user_message: str):
        """Version streaming qui retourne aussi les métadonnées"""
        try:
            # Ajoute le message utilisateur à l'historique
            self.conversation_manager.add_message("user", user_message)
            
            # Analyse la requête avec contexte
            query_analysis = self._analyze_query_with_context(user_message)
            self.conversation_manager.update_context(query_analysis)
            
            # Récupération intelligente de documents
            relevant_docs = await self._intelligent_document_retrieval(user_message, query_analysis)
            
            # Construit la requête avec contexte
            conversation_context = self.conversation_manager.get_conversation_context()
            enhanced_query = user_message
            
            if conversation_context and query_analysis.get("context_aware"):
                enhanced_query = f"{conversation_context}\n\nNouvelle question: {user_message}"
            
            if query_analysis.get("municipality"):
                municipality_info = self.document_processor.get_municipality_info(query_analysis["municipality"])
                if municipality_info:
                    context_info = f"\nInformations sur {query_analysis['municipality']}: "
                    context_info += f"Canton: {municipality_info.get('canton', 'N/A')}, "
                    context_info += f"Population: {municipality_info.get('population', 'N/A')} habitants, "
                    context_info += f"PECC: {municipality_info.get('pecc', 'Non renseigné')}"
                    enhanced_query = context_info + "\n\nQuestion: " + enhanced_query
            
            # Stream la réponse et collecte le texte complet
            response_parts = []
            async for chunk in self.mistral_client.stream_response(
                self.system_prompt,
                enhanced_query,
                relevant_docs
            ):
                response_parts.append(chunk)
                yield {
                    "type": "chunk",
                    "content": chunk
                }
            
            # Ajoute la réponse complète à l'historique
            full_response = "".join(response_parts)
            response_metadata = {
                "query_type": query_analysis["type"],
                "municipality": query_analysis.get("municipality"),
                "sector": query_analysis.get("sector"),
                "search_method": query_analysis.get("search_method", "hybrid"),
                "sources_count": len(relevant_docs)
            }
            
            self.conversation_manager.add_message("assistant", full_response, response_metadata)
            
            # Retourne les métadonnées finales
            yield {
                "type": "metadata",
                "query_analysis": query_analysis,
                "relevant_docs": relevant_docs,
                "full_response": full_response,
                "search_quality": self._assess_search_quality(relevant_docs)
            }
                
        except Exception as e:
            self.logger.error(f"Erreur lors du streaming: {e}")
            error_msg = f"Erreur: {str(e)}"
            self.conversation_manager.add_message("assistant", error_msg)
            yield {
                "type": "error",
                "content": error_msg
            }
    
    # Legacy method for backward compatibility
    async def stream_conversation(self, user_message: str):
        """Version streaming simple pour compatibilité"""
        async for data in self.stream_conversation_with_metadata(user_message):
            if data["type"] == "chunk":
                yield data["content"]
    
    def get_conversation_history(self) -> List[Dict]:
        """Retourne l'historique de conversation formaté"""
        return [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"].isoformat(),
                "metadata": msg.get("metadata", {})
            }
            for msg in self.conversation_manager.messages
        ]
    
    def get_available_cantons(self) -> List[str]:
        """Retourne les cantons disponibles"""
        return self.vector_store.get_available_cantons()

    def get_available_sectors(self) -> List[str]:
        """Retourne les secteurs disponibles"""
        return self.vector_store.get_available_sectors()

    def get_statistics(self) -> Dict:
        """Retourne des statistiques complètes"""
        vector_stats = self.vector_store.get_statistics()
        municipalities_count = len(self.document_processor.municipalities_data or {})
        conversation_stats = self.conversation_manager.get_context_summary()
        
        stats = {
            **vector_stats,
            "municipalities_loaded": municipalities_count,
            "model_info": self.mistral_client.get_model_info(),
            "conversation": conversation_stats,
            "vector_store_type": "enhanced",
            "capabilities": {
                "hybrid_search": True,
                "reranking": True,
                "query_expansion": True,
                "conversation_context": True,
                "metadata_filtering": True
            }
        }
        
        return stats
    
    def get_system_info(self) -> Dict:
        """Retourne les informations système pour debugging"""
        return {
            "vector_store_type": "enhanced",
            "vector_store_class": type(self.vector_store).__name__,
            "conversation_manager_class": type(self.conversation_manager).__name__,
            "capabilities": {
                "hybrid_search": True,
                "reranking": True,
                "query_expansion": True,
                "conversation_context": True,
                "metadata_filtering": True,
                "integrated_embeddings": True
            }
        }