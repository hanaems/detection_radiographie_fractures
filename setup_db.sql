-- Script SQL pour PostgreSQL
-- Créer la base de données (à exécuter dans pgAdmin)

CREATE DATABASE fracture_detection
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    CONNECTION LIMIT = -1;

-- Se connecter à fracture_detection puis exécuter :

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    image_filename VARCHAR(255) NOT NULL,
    model_type VARCHAR(20) NOT NULL,
    result VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2),
    has_fracture BOOLEAN,
    detections TEXT,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_analyses_user_id ON analyses(user_id);

-- Utilisateur admin par défaut (password: admin123)
INSERT INTO users (username, email, password_hash, first_name, last_name, role) 
VALUES (
    'admin',
    'admin@fracture.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYNqMU4kJFy',
    'Admin',
    'System',
    'admin'
);
