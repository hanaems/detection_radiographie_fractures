"""Script pour créer les tables"""
from database import init_db, User, SessionLocal
from auth import hash_password

print("🔧 Création des tables...")
init_db()

print("👤 Création de l'utilisateur admin...")
db = SessionLocal()

# Vérifier si admin existe déjà
existing_admin = db.query(User).filter(User.username == "admin").first()

if not existing_admin:
    admin = User(
        username="admin",
        email="admin@fracture.com",
        password_hash=hash_password("admin123"),
        first_name="Admin",
        last_name="System",
        role="admin"
    )
    db.add(admin)
    db.commit()
    print("✅ Utilisateur admin créé (username: admin, password: admin123)")
else:
    print("✅ Utilisateur admin existe déjà")

db.close()
print("\n🎉 Base de données prête !")
