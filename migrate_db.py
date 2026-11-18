"""Migration base de données - Ajout colonnes profil"""
from sqlalchemy import create_engine, text
from config import settings

engine = create_engine(settings.DATABASE_URL, echo=True)

def migrate():
    with engine.connect() as conn:
        # Ajouter les nouvelles colonnes si elles n'existent pas
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(20)"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS speciality VARCHAR(100)"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS address TEXT"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS rpps VARCHAR(50)"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS photo TEXT"))
            conn.commit()
            print("✅ Migration réussie ! Colonnes ajoutées.")
        except Exception as e:
            print(f"❌ Erreur migration: {e}")
            conn.rollback()

if __name__ == "__main__":
    migrate()
