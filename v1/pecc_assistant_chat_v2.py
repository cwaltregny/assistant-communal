import logging
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime
from process_documents import DocumentProcessor
from index_documents import VectorStore
from client_mistral import MistralClient

class ConversationManager:
    """Gère l'historique et le contexte des conversations"""
    
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
        
        if len(self.messages) > self.max_history * 2:  
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_conversation_context(self) -> str:
        """Génère un contexte de conversation pour l'IA"""
        if not self.messages:
            return ""
        
        context_parts = ["Contexte de la conversation précédente:"]
        
        recent_messages = self.messages[-6:]  
        for msg in recent_messages:
            role_fr = "Utilisateur" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_fr}: {msg['content'][:200]}...")
        
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
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.document_processor = DocumentProcessor(config)
        self.vector_store = VectorStore(config)
        self.mistral_client = MistralClient(config)
        
        self.conversation_manager = ConversationManager()
        
        self.system_prompt = """Tu es un expert en politique climatique en Suisse romande et un assistant conversationnel pour les communes suisses. Tu aides à élaborer des PECC (Plans Energie Climat Communaux) et à identifier les mesures clés pour la décarbonation et l'adaptation.

INSTRUCTIONS IMPORTANTES:
- Tu réponds toujours en français, de façon claire et structurée
- Tu maintiens une conversation fluide en te référant aux échanges précédents quand pertinent
- Tu utilises le contexte des messages précédents pour donner des réponses plus personnalisées
- Si une commune ou un secteur a été mentionné précédemment, tu t'en souviens
- Tu ne fournis jamais de lien internet
- Tu peux faire des références comme "comme nous avons discuté" ou "en lien avec votre question précédente"
- Si la question concerne une commune précise, commence par regarder le nombre d'habitants et les informations de cette commune
- Pour les questions générales sur les mesures, demande de préciser le secteur si pas mentionné
- Si tu ne trouves pas la réponse dans les documents, dis-le clairement
- Tu vérifies bien à la fin de ta réponse que tu n'as pas fourni de lien à des pages internet dans ta réponse

SECTEURS DISPONIBLES: Transport, Bâtiment, Energie, Alimentation, Aménagement du territoire"""
        
        self.secteurs_disponibles = [
            "Transport", "Bâtiment", "Energie", "Alimentation", "Aménagement du territoire"
        ]

    async def initialize(self):
        """Initialise l'assistant - charge l'index vectoriel existant"""
        self.logger.info("Initialisation de l'assistant PECC...")
        
        if not self.vector_store.load_index():
            raise ValueError(
                "Index vectoriel non trouvé. "
                "Veuillez d'abord exécuter: python generate_embeddings.py"
            )
        
        self.document_processor.load_municipalities_data()
        
        if not self.mistral_client.validate_api_connection():
            self.logger.warning("Impossible de valider la connexion à l'API Mistral")
        
        self.logger.info("Assistant PECC initialisé avec succès")

    def reset_conversation(self):
        """Démarre une nouvelle conversation"""
        self.conversation_manager.reset()
        self.logger.info("Nouvelle conversation initialisée")

    async def process_message(self, user_message: str, session_id: str = None) -> Dict:
        """Traite un message dans le contexte d'une conversation"""
        try:
            self.conversation_manager.add_message("user", user_message)
            
            query_analysis = self._analyze_query_with_context(user_message)
            
            self.conversation_manager.update_context(query_analysis)
            
            relevant_docs = await self._get_relevant_documents(user_message, query_analysis)
            
            response = await self._generate_contextual_response(
                user_message, relevant_docs, query_analysis
            )
            
            self.conversation_manager.add_message(
                "assistant", 
                response, 
                {
                    "query_type": query_analysis["type"],
                    "municipality": query_analysis.get("municipality"),
                    "sector": query_analysis.get("sector")
                }
            )
            
            references = self._format_references(relevant_docs)
            
            return {
                "response": response,
                "references": references,
                "query_type": query_analysis["type"],
                "municipality": query_analysis.get("municipality"),
                "sector": query_analysis.get("sector"),
                "conversation_context": self.conversation_manager.get_context_summary(),
                "status": "success"
            }
            
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

    def _analyze_query_with_context(self, query: str) -> Dict:
        """Analyse la requête en tenant compte du contexte de conversation"""
        query_lower = query.lower()
        
        analysis = {
            "type": "general",
            "municipality": None,
            "sector": None,
            "canton": None,
            "context_aware": False
        }
        
        current_context = self.conversation_manager.context
        
        contextual_indicators = [
            "cette commune", "cette ville", "la commune", "la ville",
            "ce secteur", "cette mesure", "comme dit", "par rapport à",
            "dans ce contexte", "pour cette situation"
        ]
        
        has_contextual_reference = any(indicator in query_lower for indicator in contextual_indicators)
        
        if has_contextual_reference:
            analysis["context_aware"] = True
            if current_context["current_municipality"] and not self._has_explicit_municipality(query):
                analysis["municipality"] = current_context["current_municipality"]
                analysis["type"] = "commune_spécifique"
            
            if current_context["current_sector"] and not self._has_explicit_sector(query):
                analysis["sector"] = current_context["current_sector"]
                if analysis["type"] == "general":
                    analysis["type"] = "secteur_spécifique"
        
        standard_analysis = self._analyze_query(query)
        
        if standard_analysis["municipality"]:
            analysis["municipality"] = standard_analysis["municipality"]
            analysis["canton"] = standard_analysis["canton"]
            analysis["type"] = standard_analysis["type"]
        
        if standard_analysis["sector"]:
            analysis["sector"] = standard_analysis["sector"]
            if analysis["type"] in ["general", "context_aware"]:
                analysis["type"] = "secteur_spécifique"
        
        if not has_contextual_reference:
            return standard_analysis
        
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

    def _analyze_query(self, query: str) -> Dict:
        """Analyse la requête pour identifier le type et extraire les entités"""
        query_lower = query.lower()
        
        analysis = {
            "type": "general",
            "municipality": None,
            "sector": None,
            "canton": None
        }
        
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
                # Valide que c'est bien une commune connue
                municipality_info = self.document_processor.get_municipality_info(potential_municipality)
                if municipality_info:
                    analysis["municipality"] = potential_municipality
                    analysis["canton"] = municipality_info.get("canton")
                    analysis["type"] = "commune_spécifique"
                    break
        
        # Détecte le secteur mentionné
        for secteur in self.secteurs_disponibles:
            if secteur.lower() in query_lower:
                analysis["sector"] = secteur
                if analysis["type"] == "general":
                    analysis["type"] = "secteur_spécifique"
                break
        
        # Détecte des mots-clés spécifiques
        if any(word in query_lower for word in ["mesure", "action", "recommandation", "priorité"]):
            if analysis["type"] == "general":
                analysis["type"] = "requête_mesures"
        
        return analysis

    async def _get_relevant_documents(self, query: str, analysis: Dict) -> List[Dict]:
        """Récupère les documents pertinents selon l'analyse de la requête"""
        relevant_docs = []
        
        # Si c'est une question spécifique à une commune
        if analysis["type"] == "commune_spécifique" and analysis.get("canton"):
            # Recherche dans les plans climat du canton
            canton_docs = self.vector_store.search_by_canton(query, analysis["canton"], k=3)
            relevant_docs.extend(canton_docs)
            
            # Ajoute aussi des mesures générales pertinentes
            general_measures = self.vector_store.search_by_document_type(query, "mesure_prioritaire", k=3)
            relevant_docs.extend(general_measures)
        
        # Si c'est une question spécifique à un secteur
        elif analysis["type"] == "secteur_spécifique" and analysis.get("sector"):
            sector_docs = self.vector_store.search_by_sector(query, analysis["sector"], k=5)
            relevant_docs.extend(sector_docs)
        
        # Si c'est une demande de mesures générales
        elif analysis["type"] == "requête_mesures":
            measures_docs = self.vector_store.search_by_document_type(query, "mesure_prioritaire", k=5)
            relevant_docs.extend(measures_docs)
        
        # Recherche générale
        else:
            general_docs = self.vector_store.search(query, k=5)
            relevant_docs.extend(general_docs)
        
        # Supprime les doublons et trie par score
        unique_docs = {}
        for doc in relevant_docs:
            chunk_id = doc["chunk_id"]
            if chunk_id not in unique_docs or doc["similarity_score"] > unique_docs[chunk_id]["similarity_score"]:
                unique_docs[chunk_id] = doc
        
        return sorted(unique_docs.values(), key=lambda x: x["similarity_score"], reverse=True)[:5]

    async def _generate_contextual_response(self, query: str, relevant_docs: List[Dict], analysis: Dict) -> str:
        """Génère la réponse en tenant compte du contexte de conversation"""
        
        # Construit le contexte de conversation
        conversation_context = self.conversation_manager.get_conversation_context()
        
        # Enrichit la requête avec des informations contextuelles
        enhanced_query = query
        
        # Ajoute le contexte de conversation si pertinent
        if conversation_context and analysis.get("context_aware"):
            enhanced_query = f"{conversation_context}\n\nNouvelle question: {query}"
        
        # Si c'est une question sur une commune, ajoute les infos de la commune
        if analysis.get("municipality"):
            municipality_info = self.document_processor.get_municipality_info(analysis["municipality"])
            if municipality_info:
                context_info = f"\nInformations sur {analysis['municipality']}: "
                context_info += f"Canton: {municipality_info.get('canton', 'N/A')}, "
                context_info += f"Population: {municipality_info.get('population', 'N/A')} habitants, "
                context_info += f"Superficie: {municipality_info.get('superficie', 'N/A')} km², "
                context_info += f"Densité: {municipality_info.get('densite', 'N/A')} hab/km², "
                context_info += f"PECC existant: {municipality_info.get('pecc', 'Non renseigné')}"
                
                # Ajoute les détails si disponibles
                details = municipality_info.get('details', '')
                if details and len(details) > 50:
                    context_info += f"\nDétails sur les actions climat de la commune: {details[:500]}..."
                
                enhanced_query = context_info + "\n\nQuestion: " + enhanced_query
        
        # Si pas de secteur spécifié pour une demande de mesures, demande clarification
        if analysis["type"] == "requête_mesures" and not analysis.get("sector") and not analysis.get("municipality"):
            sectors_list = ", ".join(self.secteurs_disponibles)
            clarification_prompt = f"""L'utilisateur demande des informations sur les mesures prioritaires mais n'a pas spécifié de secteur. 
            
Demande-lui de préciser sur quel secteur il souhaite se concentrer parmi: {sectors_list}.
            
Question originale: {query}"""
            
            return await self.mistral_client.generate_response(
                self.system_prompt,
                clarification_prompt,
                []
            )
        
        # Génère la réponse normale
        return await self.mistral_client.generate_response(
            self.system_prompt,
            enhanced_query,
            relevant_docs
        )

    def _format_references(self, documents: List[Dict]) -> List[Dict]:
        """Formate les références des documents utilisés"""
        references = []
    
        for doc in documents:
            # Limite la taille de l'extrait selon le contexte
            content = doc["content"]
            if len(content) > 500:
                # Trouve une coupure naturelle près de 400 caractères
                cut_point = content.find(' ', 350)
                if cut_point == -1 or cut_point > 500:
                    cut_point = 400
                extract = content[:cut_point] + "..."
            else:
                extract = content
            
            ref = {
                "source_file": doc["source_file"],
                "page_number": doc["page_number"],
                "document_type": doc["document_type"],
                "similarity_score": round(doc["similarity_score"], 3),
                "extract": extract, 
                "full_content": content  
            }
            
            # Ajoute les métadonnées spécifiques de façon plus efficace
            metadata = doc.get("metadata", {})
            if doc["document_type"] == "plan_climat":
                ref["canton"] = metadata.get("canton")
            elif doc["document_type"] == "mesure_prioritaire":
                ref["secteur"] = metadata.get("secteur")
            
            references.append(ref)
        
        return references

    async def stream_conversation_with_metadata(self, user_message: str):
        """Version streaming qui retourne aussi les métadonnées"""
        try:
            # Ajoute le message utilisateur à l'historique
            self.conversation_manager.add_message("user", user_message)
            
            query_analysis = self._analyze_query_with_context(user_message)
            self.conversation_manager.update_context(query_analysis)
            
            relevant_docs = await self._get_relevant_documents(user_message, query_analysis)
            
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
            self.conversation_manager.add_message("assistant", full_response)
            
            # Retourne les métadonnées finales
            yield {
                "type": "metadata",
                "query_analysis": query_analysis,
                "relevant_docs": relevant_docs,
                "full_response": full_response
            }
                
        except Exception as e:
            self.logger.error(f"Erreur lors du streaming: {e}")
            error_msg = f"Erreur: {str(e)}"
            self.conversation_manager.add_message("assistant", error_msg)
            yield {
                "type": "error",
                "content": error_msg
            }

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
        """Retourne des statistiques sur les documents et la conversation"""
        vector_stats = self.vector_store.get_statistics()
        municipalities_count = len(self.document_processor.municipalities_data or {})
        conversation_stats = self.conversation_manager.get_context_summary()
        
        return {
            **vector_stats,
            "municipalities_loaded": municipalities_count,
            "model_info": self.mistral_client.get_model_info(),
            "conversation": conversation_stats
        }

    async def refresh_index(self):
        """Reconstruit l'index depuis les fichiers"""
        self.logger.info("Reconstruction de l'index...")
        await self._build_new_index()
        self.logger.info("Index reconstruit avec succès")

    async def _build_new_index(self):
        """Construit un nouvel index vectoriel"""
        # Traite tous les documents
        chunks = self.document_processor.process_all_documents()
        
        if not chunks:
            raise ValueError("Aucun document n'a pu être traité")
        
        # Construit l'index
        self.vector_store.build_index(chunks)
        
        # Sauvegarde l'index
        self.vector_store.save_index()