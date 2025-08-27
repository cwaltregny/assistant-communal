import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    MISTRAL_MODEL: str = "mistral-medium-latest"
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    
    PLAN_CLIMAT_FOLDER: str = "./data/plan_climat_cantonaux"
    PRIORITY_MEASURES_FOLDER: str = "./data/mesures_prioritaires"
    MUNICIPALITIES_CSV: str = "./data/donnees_communes_clean.csv"
    
    VECTOR_DB_PATH: str = "./vector_db"
    EMBEDDINGS_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    
    class Config:
        env_file = ".env"

settings = Settings()