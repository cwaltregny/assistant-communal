import streamlit as st
import asyncio
import pandas as pd
from typing import Dict, List
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import uuid

sys.path.append(str(Path(__file__).parent))

from config_ac import settings
from pecc_assistant_hybrid import PECCAssistant

st.set_page_config(
    page_title="Assistant PECC",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #f8f9fa;
    }
    .user-msg {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 20%;
    }
    .assistant-msg {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        margin-right: 20%;
    }
    .context-info {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .quality-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem 0;
    }
    .quality-excellent { background-color: #c8e6c9; color: #2e7d32; }
    .quality-good { background-color: #dcedc8; color: #558b2f; }
    .quality-fair { background-color: #fff3e0; color: #f57c00; }
    .quality-poor { background-color: #ffcdd2; color: #d32f2f; }
    .reference-summary {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .reference-header {
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.25rem;
    }
    .reference-meta {
        color: #666;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session():
    """Initialise la session"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

@st.cache_resource
def load_assistant():
    """Charge l'assistant PECC hybride"""
    try:
        assistant = PECCAssistant(settings)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(assistant.initialize())
        loop.close()
        return assistant, None
    except Exception as e:
        return None, str(e)

def display_message(role: str, content: str, timestamp: str = None, metadata: Dict = None):
    """Affiche un message dans le chat"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    css_class = "user-msg" if role == "user" else "assistant-msg"
    role_name = "Vous" if role == "user" else "Assistant PECC"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{role_name}:</strong> {content}
        <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if role == "assistant" and metadata:
        context_parts = []
        if metadata.get('municipality'):
            context_parts.append(f"📍 {metadata['municipality']}")
        if metadata.get('sector'):
            context_parts.append(f"🎯 {metadata['sector']}")
        if metadata.get('query_type'):
            context_parts.append(f"📋 {metadata['query_type']}")
        if metadata.get('search_method'):
            context_parts.append(f"🔍 {metadata['search_method']}")
        
        if context_parts:
            context_text = " | ".join(context_parts)
            st.markdown(f"""
            <div class="context-info">
                <small>{context_text}</small>
            </div>
            """, unsafe_allow_html=True)

def display_search_quality(search_quality: Dict):
    """Affiche la qualité de la recherche"""
    if not search_quality:
        return
    
    quality = search_quality.get('quality', 'unknown')
    quality_colors = {
        'excellent': 'quality-excellent',
        'good': 'quality-good', 
        'fair': 'quality-fair',
        'poor': 'quality-poor'
    }
    
    quality_class = quality_colors.get(quality, 'quality-fair')
    
    st.markdown(f"""
    <div class="quality-indicator {quality_class}">
        Qualité de recherche: {quality.upper()} 
        (Score max: {search_quality.get('max_score', 0):.3f}, 
        {search_quality.get('num_results', 0)} résultats)
    </div>
    """, unsafe_allow_html=True)

def display_references_optimized(references: List[Dict], search_quality: Dict = None):
    """Affiche les références de manière optimisée"""
    if not references:
        return
    
    sources_summary = []
    for ref in references[:3]:
        filename = ref['source_file'].split('/')[-1]
        sources_summary.append(f"{filename} (p.{ref['page_number']})")
    
    summary_text = " • ".join(sources_summary)
    if len(references) > 3:
        summary_text += f" • +{len(references)-3} autres"
    
    with st.expander(f"📚 Sources: {summary_text}"):
        if search_quality:
            display_search_quality(search_quality)
        
        for i, ref in enumerate(references, 1):
            advanced_info = []
            if ref.get('semantic_score'):
                advanced_info.append(f"Sémantique: {ref['semantic_score']:.3f}")
            if ref.get('lexical_score'):
                advanced_info.append(f"Lexical: {ref['lexical_score']:.3f}")
            if ref.get('rerank_score'):
                advanced_info.append(f"Rerank: {ref['rerank_score']:.3f}")
            if ref.get('search_method'):
                advanced_info.append(f"Méthode: {ref['search_method']}")
            
            advanced_text = " | ".join(advanced_info) if advanced_info else ""
            
            st.markdown(f"""
            **📄 {ref['source_file']}** (page {ref['page_number']}) - Pertinence: {ref['similarity_score']:.3f}
            
            {f"*{advanced_text}*" if advanced_text else ""}
            
            {ref['extract']}
            """)
        

async def process_message_async(assistant, message: str):
    """Traite un message de façon asynchrone"""
    return await assistant.process_message(message)

def main():
    init_session()
    
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Assistant PECC</h1>
        <p>Expert en Plans Energie Climat Communaux</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.assistant is None:
        with st.spinner("🔄 Initialisation de l'assistant..."):
            assistant, error = load_assistant()
            if assistant:
                st.session_state.assistant = assistant
                st.success("✅ Assistant initialisé!")
                
                system_info = assistant.get_system_info()
                capabilities = system_info.get('capabilities', {})
                cap_text = []
                if capabilities.get('hybrid_search'):
                    cap_text.append("🔍 Recherche Hybride")
                if capabilities.get('reranking'):
                    cap_text.append("🎯 Reranking")
                
                if cap_text:
                    st.info(f"**Capacités activées:** {' | '.join(cap_text)}")
                    
            else:
                st.error(f"❌ Erreur: {error}")
                st.error("🔧 Vérifiez que vous avez:")
                st.error("   1. Exécuté le script de génération d'embeddings")
                st.error("   2. Créé le fichier enhanced_vector_store.py")
                st.error("   3. Installé les dépendances requises")
                st.stop()

    assistant = st.session_state.assistant

    with st.sidebar:
        st.header("📊 Informations")
        
        try:
            stats = assistant.get_statistics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", f"{stats.get('total_chunks', 0):,}")
            with col2:
                st.metric("Communes", stats.get('municipalities_loaded', 0))
            
            st.markdown("**🚀 Capacités:**")
            capabilities = stats.get('capabilities', {})
            for cap, enabled in capabilities.items():
                icon = "✅" if enabled else "❌"
                cap_name = cap.replace('_', ' ').title()
                st.write(f"{icon} {cap_name}")
            
            conv_stats = stats.get('conversation', {})
            if conv_stats.get('message_count', 0) > 3:
                st.markdown("**💬 Conversation**")
                st.write(f"Messages: {conv_stats.get('message_count', 0)}")
                if conv_stats.get('current_municipality'):
                    st.write(f"Commune: {conv_stats['current_municipality']}")
                if conv_stats.get('current_sector'):
                    st.write(f"Secteur: {conv_stats['current_sector']}")
                    
        except Exception as e:
            st.error(f"Erreur stats: {e}")

        st.markdown("---")
        
        if st.button("🔄 Nouvelle conversation"):
            assistant.reset_conversation()
            st.session_state.messages = []
            st.rerun()

        response_mode = st.radio("Mode de réponse", ["Streaming","Standard"])
        
        st.markdown("---")
        
        with st.expander("🔧 Info Système"):
            system_info = assistant.get_system_info()
            st.json(system_info)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat")
        
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background-color: #f5f5f5; border-radius: 0.5rem;">
                    💬 <strong>Commencez une conversation!</strong><br>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    display_message(
                        msg['role'], 
                        msg['content'], 
                        msg.get('timestamp'), 
                        msg.get('metadata')
                    )
                    
                    if msg['role'] == 'assistant' and 'references' in msg:
                        display_references_optimized(
                            msg['references'], 
                            msg.get('search_quality')
                        )

        st.markdown("---")
        
        user_input = st.chat_input("Tapez votre message...")
        
        if user_input:
            user_msg = {
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().strftime("%H:%M")
            }
            st.session_state.messages.append(user_msg)
            
            with st.spinner("💭 L'assistant réfléchit..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    if response_mode == "Streaming":
                        response_placeholder = st.empty()
                        
                        response_text = ""
                        query_analysis = None
                        relevant_docs = []
                        search_quality = {}
                        
                        async def stream_with_metadata():
                            nonlocal response_text, query_analysis, relevant_docs, search_quality
                            async for data in assistant.stream_conversation_with_metadata(user_input):
                                if data["type"] == "chunk":
                                    response_text += data["content"]
                                    response_placeholder.markdown(f"""
                                    <div class="chat-message assistant-msg">
                                        <strong>Assistant PECC:</strong> {response_text}
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif data["type"] == "metadata":
                                    query_analysis = data["query_analysis"]
                                    relevant_docs = data["relevant_docs"]
                                    search_quality = data.get("search_quality", {})
                                elif data["type"] == "error":
                                    response_text = data["content"]
                                    response_placeholder.markdown(f"""
                                    <div class="chat-message assistant-msg">
                                        <strong>Assistant PECC:</strong> ❌ {response_text}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        loop.run_until_complete(stream_with_metadata())
                        
                        references = assistant._format_references(relevant_docs) if relevant_docs else []
                        
                        assistant_msg = {
                            'role': 'assistant',
                            'content': response_text,
                            'timestamp': datetime.now().strftime("%H:%M"),
                            'metadata': {
                                'query_type': query_analysis.get('type') if query_analysis else None,
                                'municipality': query_analysis.get('municipality') if query_analysis else None,
                                'sector': query_analysis.get('sector') if query_analysis else None,
                                'search_method': query_analysis.get('search_method') if query_analysis else None
                            },
                            'references': references,
                            'search_quality': search_quality
                        }
                        
                    else:
                        result = loop.run_until_complete(process_message_async(assistant, user_input))
                        
                        if result['status'] == 'success':
                            assistant_msg = {
                                'role': 'assistant',
                                'content': result['response'],
                                'timestamp': datetime.now().strftime("%H:%M"),
                                'metadata': {
                                    'query_type': result.get('query_type'),
                                    'municipality': result.get('municipality'),
                                    'sector': result.get('sector'),
                                    'search_method': result.get('search_method')
                                },
                                'references': result.get('references', []),
                                'search_quality': result.get('search_quality', {})
                            }
                        else:
                            assistant_msg = {
                                'role': 'assistant',
                                'content': f"❌ Erreur: {result.get('error', 'Erreur inconnue')}",
                                'timestamp': datetime.now().strftime("%H:%M")
                            }
                    
                    st.session_state.messages.append(assistant_msg)
                    
                except Exception as e:
                    error_msg = {
                        'role': 'assistant',
                        'content': f"❌ Erreur lors du traitement: {str(e)}",
                        'timestamp': datetime.now().strftime("%H:%M")
                    }
                    st.session_state.messages.append(error_msg)
                
                finally:
                    loop.close()
            
            st.rerun()

    with col2:
        st.header("💡 Aide")
        
        st.markdown("**🚀 Exemples de questions:**")
        
        examples = [
            "Comment créer un PECC?",
            "Je fais partie de la commune de Choulex, peux-tu m'aider à élaborer un PECC en mettant en avant les mesures prioritaires?",
            "Quelles sont les mesures prioritaires pour le transport durable?",
            "Je fais partie de la commune de Satigny, quelles subventions puis-je obtenir du canton pour la rénovation énergétique des mes bâtiments publics?"
        ]
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"ex_{hash(example)}", use_container_width=True):
                user_msg = {
                    'role': 'user',
                    'content': example,
                    'timestamp': datetime.now().strftime("%H:%M")
                }
                st.session_state.messages.append(user_msg)
                
                with st.spinner("💭 Traitement..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        result = loop.run_until_complete(process_message_async(assistant, example))
                        
                        if result['status'] == 'success':
                            assistant_msg = {
                                'role': 'assistant',
                                'content': result['response'],
                                'timestamp': datetime.now().strftime("%H:%M"),
                                'metadata': {
                                    'query_type': result.get('query_type'),
                                    'municipality': result.get('municipality'),
                                    'sector': result.get('sector'),
                                    'search_method': result.get('search_method')
                                },
                                'references': result.get('references', []),
                                'search_quality': result.get('search_quality', {})
                            }
                        else:
                            assistant_msg = {
                                'role': 'assistant',
                                'content': f"❌ Erreur: {result.get('error')}",
                                'timestamp': datetime.now().strftime("%H:%M")
                            }
                        
                        st.session_state.messages.append(assistant_msg)
                        
                    except Exception as e:
                        error_msg = {
                            'role': 'assistant',
                            'content': f"❌ Erreur: {str(e)}",
                            'timestamp': datetime.now().strftime("%H:%M")
                        }
                        st.session_state.messages.append(error_msg)
                    
                    finally:
                        loop.close()
                
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        **💡 Conseils:**
        
        • **Soyez spécifique** dans vos questions
        
        • **Mentionnez votre commune** pour des conseils personnalisés
        
        • **Précisez le secteur** d'intérêt
        
        • **Référez-vous aux échanges précédents** avec "cette commune", "ce secteur"
        
        • **La recherche hybride** combine sémantique et lexical pour de meilleurs résultats
        
        """)
        
        st.markdown("---")
        
        if st.session_state.messages:
            st.markdown("**📤 Export**")
            
            if st.button("💾 Télécharger conversation"):
                export_content = "# Conversation PECC Assistant\n\n"
                export_content += f"**Date:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
                
                for msg in st.session_state.messages:
                    role = "**Vous**" if msg['role'] == 'user' else "**Assistant PECC**"
                    timestamp = msg.get('timestamp', '')
                    export_content += f"## {role} ({timestamp})\n\n{msg['content']}\n\n"
                    
                    if msg.get('metadata'):
                        metadata = msg['metadata']
                        export_content += f"*Métadonnées: {metadata}*\n\n"
                
                st.download_button(
                    "📥 Télécharger",
                    export_content,
                    file_name=f"conversation_pecc_hybride_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )

def show_statistics():
    """Affiche les statistiques détaillées"""
    if 'assistant' not in st.session_state or st.session_state.assistant is None:
        st.warning("Assistant non initialisé")
        return
    
    assistant = st.session_state.assistant
    
    st.header("📊 Statistiques détaillées")
    
    try:
        stats = assistant.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents indexés", f"{stats.get('total_chunks', 0):,}")
        
        with col2:
            st.metric("Communes chargées", stats.get('municipalities_loaded', 0))
        
        with col3:
            conv_stats = stats.get('conversation', {})
            st.metric("Messages conversation", conv_stats.get('message_count', 0))
        
        with col4:
            st.metric("Type Vector Store", stats.get('vector_store_type', 'N/A'))
        
        st.subheader("🚀 Capacités du système")
        capabilities = stats.get('capabilities', {})
        
        cap_cols = st.columns(3)
        cap_items = list(capabilities.items())
        
        for i, (cap, enabled) in enumerate(cap_items):
            with cap_cols[i % 3]:
                icon = "✅" if enabled else "❌"
                cap_name = cap.replace('_', ' ').title()
                st.write(f"{icon} **{cap_name}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Cantons disponibles")
            cantons = assistant.get_available_cantons()
            for canton in cantons:
                st.write(f"• {canton}")
        
        with col2:
            st.subheader("🏭 Secteurs disponibles")
            sectors = assistant.get_available_sectors()
            for sector in sectors:
                st.write(f"• {sector}")
        
        if conv_stats.get('message_count', 0) > 0:
            st.subheader("💬 Analyse de conversation")
            
            user_messages = [msg for msg in st.session_state.messages if msg['role'] == 'user']
            assistant_messages = [msg for msg in st.session_state.messages if msg['role'] == 'assistant']
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.write(f"**Messages utilisateur:** {len(user_messages)}")
                st.write(f"**Réponses assistant:** {len(assistant_messages)}")
                
                if user_messages:
                    avg_user_length = sum(len(msg['content']) for msg in user_messages) / len(user_messages)
                    st.write(f"**Longueur moyenne (utilisateur):** {avg_user_length:.0f} caractères")
            
            with analysis_col2:
                search_qualities = []
                search_methods = []
                
                for msg in assistant_messages:
                    if 'search_quality' in msg:
                        sq = msg['search_quality']
                        if sq.get('quality'):
                            search_qualities.append(sq['quality'])
                    
                    if 'metadata' in msg and msg['metadata'].get('search_method'):
                        search_methods.append(msg['metadata']['search_method'])
                
                if search_qualities:
                    quality_counts = {q: search_qualities.count(q) for q in set(search_qualities)}
                    st.write("**Qualités de recherche:**")
                    for quality, count in quality_counts.items():
                        st.write(f"• {quality}: {count}")
                
                if search_methods:
                    method_counts = {m: search_methods.count(m) for m in set(search_methods)}
                    st.write("**Méthodes utilisées:**")
                    for method, count in method_counts.items():
                        st.write(f"• {method}: {count}")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des statistiques: {e}")

def main_with_tabs():
    """Interface principale avec onglets"""
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Statistiques"])
    
    with tab1:
        main()
    
    with tab2:
        show_statistics()

if __name__ == "__main__":
    main_with_tabs()