"""Gestion base de données PostgreSQL"""
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config import settings

Base = declarative_base()

# Table Users
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    phone = Column(String(20))
    speciality = Column(String(100))
    address = Column(Text)
    rpps = Column(String(50))
    photo = Column(Text)  # base64 encoded image
    role = Column(String(20), default="doctor")  # doctor, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    analyses = relationship("Analysis", back_populates="user")

# Table Analyses
class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_filename = Column(String(255), nullable=False)
    model_type = Column(String(20), nullable=False)  # efficientnet, yolo, combined
    result = Column(String(50), nullable=False)
    confidence = Column(Float)
    has_fracture = Column(Boolean)
    detections = Column(Text)  # JSON string
    analysis_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    user = relationship("User", back_populates="analyses")

# Connexion
engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Créer toutes les tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées!")

def get_db():
    """Obtenir session DB"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
