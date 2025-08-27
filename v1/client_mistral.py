"""
mistral_client.py
"""
import httpx
import json
import logging
from typing import List, Dict, Optional, AsyncGenerator
import asyncio

class MistralClient:
    def __init__(self, config):
        self.config = config
        self.api_key = config.MISTRAL_API_KEY
        self.base_url = config.MISTRAL_BASE_URL
        self.model = config.MISTRAL_MODEL
        self.logger = logging.getLogger(__name__)
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate_response(
        self, 
        system_prompt: str, 
        user_message: str, 
        context_documents: List[Dict] = None,
        max_tokens: int = None,
        temperature: float = 0.1
    ) -> str:
        """Génère une réponse avec Mistral"""
        try:
            # Construit le prompt avec contexte
            full_prompt = self._build_context_prompt(system_prompt, user_message, context_documents)
            
            # Prépare la requête
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.config.MAX_TOKENS,
                "temperature": temperature,
                "top_p": 0.9
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                else:
                    self.logger.error(f"Réponse inattendue de Mistral: {result}")
                    return "Désolé, je n'ai pas pu générer une réponse."
                    
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Erreur HTTP Mistral ({e.response.status_code}): {e.response.text}")
            return f"Erreur de communication avec l'API Mistral: {e.response.status_code}"
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de réponse: {e}")
            return "Désolé, une erreur s'est produite lors de la génération de la réponse."

    def _build_context_prompt(
        self, 
        system_prompt: str, 
        user_message: str, 
        context_documents: List[Dict] = None
    ) -> str:
        """Construit le prompt avec le contexte des documents"""
        
        if not context_documents:
            return user_message
        
        # Trie les documents par score de similarité
        context_documents = sorted(
            context_documents, 
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )
        
        context_parts = []
        context_parts.append("=== CONTEXTE DOCUMENTAIRE ===\n")
        
        for i, doc in enumerate(context_documents[:5]):  # Limite à 5 documents
            source_info = f"Document: {doc['source_file']}, Page: {doc['page_number']}"
            
            # Ajoute des informations spécifiques selon le type de document
            if doc['document_type'] == 'plan_climat':
                canton = doc.get('metadata', {}).get('canton', 'Inconnu')
                source_info += f", Canton: {canton}"
            elif doc['document_type'] == 'mesure_prioritaire':
                secteur = doc.get('metadata', {}).get('secteur', 'Inconnu')
                source_info += f", Secteur: {secteur}"
            
            context_parts.append(f"[EXTRAIT {i+1}]")
            context_parts.append(f"Source: {source_info}")
            context_parts.append(f"Score de pertinence: {doc['similarity_score']:.3f}")
            context_parts.append(f"Contenu: {doc['content']}\n")
        
        context_parts.append("=== FIN DU CONTEXTE ===\n")
        context_parts.append(f"Question de l'utilisateur: {user_message}")
        
        return "\n".join(context_parts)

    async def stream_response(
        self, 
        system_prompt: str, 
        user_message: str, 
        context_documents: List[Dict] = None,
        max_tokens: int = None,
        temperature: float = 0.1
    ) -> AsyncGenerator[str, None]:
        """Version streaming de la génération de réponse"""
        try:
            full_prompt = self._build_context_prompt(system_prompt, user_message, context_documents)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.config.MAX_TOKENS,
                "temperature": temperature,
                "stream": True
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: "
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                json_data = json.loads(data)
                                if "choices" in json_data and json_data["choices"]:
                                    delta = json_data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            self.logger.error(f"Erreur lors du streaming: {e}")
            yield f"Erreur: {str(e)}"

    def validate_api_connection(self) -> bool:
        """Teste la connexion à l'API Mistral"""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                return response.status_code == 200
                
        except Exception as e:
            self.logger.error(f"Validation API échouée: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Récupère des informations sur le modèle"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.config.MAX_TOKENS
        }