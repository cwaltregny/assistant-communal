# PECC Assistant - Climate Action Planning for Swiss Municipalities

## Project Overview

The PECC Assistant is an advanced Retrieval-Augmented Generation (RAG) system specifically designed to support Swiss municipalities in developing and implementing Climate and Energy Action Plans (Plans Énergie Climat Communaux - PECC). This specialized AI assistant leverages comprehensive climate policy documents from Swiss cantons and priority action frameworks to provide contextual, location-specific guidance for local climate initiatives.

## Domain-Specific Focus

This system addresses the critical need for accessible climate action guidance in Swiss municipalities. With Switzerland's ambitious goal of achieving net-zero emissions by 2050, local governments require expert knowledge to develop effective climate strategies. The PECC Assistant bridges this gap by making specialized climate policy knowledge accessible through an intelligent conversational interface.

## Key Features

### RAG Pipeline

- **Hybrid Search**: Combines semantic and lexical search techniques for optimal document retrieval
- **Reranking**: Uses cross-encoder models to impove result relevance and ranking
- **Streaming Responses**: Real-time response generation for enhanced user experience
- **Metadata Filtering**: Filtering by canton, municipality, sector, and document type
- **Conversation Context**: Maintains conversation history for contextually aware responses

### Data Sources and Processing

The system processes multiple specialized data sources:

1. **Cantonal Climate Plans (Plans Climat Cantonaux)**:
   - Geneva cantonal climate plan
   - Vaud cantonal climate plan  
   - Neuchâtel cantonal climate plan
   - Fribourg cantonal climate plan
   - Valais cantonal climate plan

2. **Priority Action created by NGO Shift Ta Commune**:
   - Transport and mobility measures
   - Building renovation strategies
   - Energy transition guidelines
   - Sustainable food systems
   - Territorial planning approaches
   - Wood construction practices

3. **Municipal Database**:
   - Comprehensive data on 2,100+ Swiss municipalities
   - Population, area, density metrics
   - Existing PECC status tracking
   - Canton affiliations

### Technical Architecture

- **Enhanced Vector Store**: Custom implementation with FAISS indexing and hybrid search capabilities
- **Multilingual Embeddings**: French optimized sentence transformers for accurate semantic understanding
- **Document Chunking**: Text segmentation preserving document context (note that due to very limited memory resources chunk sizes are very small)
- **Metadata Enrichment**: Automatic extraction of canton, sector, and document type information

## Installation and Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit`: Web interface
- `sentence-transformers`: Multilingual embeddings
- `faiss-cpu`: Vector similarity search
- `scikit-learn`: Text processing and hybrid search
- `PyMuPDF`: PDF document processing
- `mistral-client`: Language model integration

### Configuration

1. Create a `.env` file with your API credentials:
```bash
MISTRAL_API_KEY=your_mistral_api_key
```

2. Ensure data directories exist:
```
data/
├── plan_climat_cantonaux/     # Cantonal climate plans (PDFs)
├── mesures_prioritaires/      # Priority measures (PDFs) 
└── donnees_communes_clean.csv # Municipal database
```

### Setup Process

1. **Generate Vector Index**:
The vector index is already generated but in case you cannot download the DB for some reason
```bash
python generate_embeddings_enhanced.py
```

2. **Launch Application**:
```bash
streamlit run main_chat_hybrid.py
```

## Usage Examples

### Municipal-Specific Queries
```
"Je fais partie de la commune de Choulex, peux-tu m'aider à élaborer un PECC?"
"Quelles subventions sont disponibles dans le canton de Genève pour la rénovation énergétique?"
```

### Sector-Specific Questions
```
"Quelles sont les mesures prioritaires pour le transport durable?"
"Comment développer l'alimentation durable dans ma commune?"
```

### General Climate Planning
```
"Comment créer un PECC efficace?"
"Quels indicateurs utiliser pour suivre les progrès climatiques?"
```

## RAG Evaluation Framework

### Evaluation Methodology

For the evaluation, there is 2 layers. The `evaluation_pipeline.py` script that runs and generate the metrics for the evaluation of an individual PECCAssistant (either hybrid or simple). Then as I was on the march to improve the v1, I included an hybrid and reranking and wanted to compare the 2 PECCAssistants. So the `generate_questions_pairs.py` is used to create the ground truth dataset (using OpenAI) and the `evaluate_rag_with_gt_dataset.py` is used to compare both assistants on the same ground truth dataset. 

#### Evaluation Pipeline

- **Hit Rate@k**: Percentage of queries where relevant documents appear in top-k results
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant result
- **Normalized Discounted Cumulative Gain (NDCG)**: Quality-weighted ranking measure
- **Precision@k**: Proportion of relevant documents in top-k results
- **F1-Score**: Harmonic mean of precision and recall

### Evaluation Results

#### System Comparison

##### Standard Search Method

| k | Hit Rate | MRR | NDCG | Precision@K | Recall@K | Avg Retrieval Time (s) | Overall Score |
|---|----------|-----|------|-------------|----------|------------------------|---------------|
| 10 | 0.355 | 0.475 | 0.361 | 0.140 | 0.355 | 0.013 | 0.397 |
| 7 | 0.310 | 0.470 | 0.354 | 0.154 | 0.310 | 0.013 | 0.378 |
| 5 | 0.284 | 0.470 | 0.362 | 0.176 | 0.284 | 0.013 | 0.372 |
| 3 | 0.241 | 0.460 | 0.378 | 0.193 | 0.241 | 0.019 | 0.360 |

##### Advanced Search Method (Hybrid + Reranking)

| k | Hit Rate | MRR | NDCG | Precision@K | Recall@K | Avg Retrieval Time (s) | Overall Score |
|---|----------|-----|------|-------------|----------|------------------------|---------------|
| 10 | **0.540** | **0.599** | **0.530** | 0.185 | **0.540** | 0.249 | **0.556** |
| 7 | 0.504 | 0.577 | 0.504 | 0.189 | 0.504 | 0.213 | 0.528 |
| 3 | 0.450 | **0.617** | **0.516** | **0.220** | 0.450 | 0.150 | 0.527 |
| 5 | 0.476 | 0.593 | 0.498 | 0.194 | 0.476 | 0.180 | 0.522 |

### Running Evaluations

Execute the evaluation with:
```bash
python evaluation_pipeline.py 
```
for the individual evaluation and with
```bash
python evaluation_rag_with_gt_dataset.py --quick-test
```
for the comparison between assistants using the Open AI generated dataset

## System Capabilities

### Streaming Responses
Real-time response generation with progressive content delivery and metadata tracking throughout the conversation flow.

### Hybrid Search Implementation
Advanced retrieval combining:
- **Semantic Search**: Dense vector similarity using multilingual sentence transformers
- **Lexical Search**: TF-IDF based keyword matching with n-gram analysis
- **Score Fusion**: Weighted combination optimizing for both precision and recall

### Reranking Pipeline
Cross-encoder reranking using `ms-marco-MiniLM-L-6-v2` to improve relevance ordering of retrieved documents.

### Metadata Filtering
Sophisticated filtering capabilities:
- **Canton-based filtering**: Target specific cantonal jurisdictions
- **Sector-based filtering**: Focus on transport, energy, buildings, food, or territorial planning
- **Document type filtering**: Distinguish between cantonal plans and priority measures
- **Municipal context**: Automatic population and geographic context integration

### Advanced Data Curation

#### PDF Processing Pipeline
- Text extraction from multi-page climate policy documents
- Context-preserving chunking with overlap management
- Automatic metadata extraction and enrichment
- Quality filtering and validation

#### Municipal Database Integration
Comprehensive Swiss municipal database (gathered using deep research on OpenAI) featuring:
- 2,100+ municipalities with population, area, density
- Canton affiliations and administrative boundaries
- Existing PECC implementation status
- Geographic and demographic context for tailored recommendations

## Architecture Details

### Enhanced Vector Store
Custom implementation featuring:
- **Hybrid indexing**: Combined dense and sparse representations
- **Query expansion**: Automatic query augmentation with domain synonyms
- **Dynamic reranking**: Context-aware result optimization
- **Conversation context**: Multi-turn dialogue awareness

### Quality Assurance
- **Search quality assessment**: Automatic relevance scoring
- **Source diversity**: Multi-document evidence aggregation
- **Confidence indicators**: Uncertainty quantification and reporting

