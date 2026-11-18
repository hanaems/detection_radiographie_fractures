import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService } from '../services/auth.service';
import { SidebarComponent } from '../shared/sidebar/sidebar';

@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  templateUrl: './profile.html',
  styleUrls: ['./profile.scss']
})
export class ProfileComponent implements OnInit {
  profile = {
    fullName: '',
    email: '',
    phone: '',
    speciality: '',
    address: '',
    rpps: '',
    photo: null as string | null
  };

  photoPreview: string | null = null;
  currentPassword = '';
  newPassword = '';
  confirmPassword = '';
  editMode = false;
  showCurrentPassword = false;
  userId: number | null = null;

  constructor(
    private router: Router,
    private authService: AuthService
  ) {}

  ngOnInit() {
    // Récupérer l'ID utilisateur
    const userIdStr = localStorage.getItem('userId');
    if (!userIdStr) {
      alert('Session expirée, veuillez vous reconnecter');
      this.router.navigate(['/auth/login']);
      return;
    }
    
    this.userId = parseInt(userIdStr);
    this.loadProfile();
  }

  loadProfile() {
    if (!this.userId) return;

    this.authService.getProfile(this.userId).subscribe({
      next: (data) => {
        this.profile = {
          fullName: data.full_name || '',
          email: data.email || '',
          phone: data.phone || '',
          speciality: data.speciality || '',
          address: data.address || '',
          rpps: data.rpps || '',
          photo: data.photo
        };
        this.photoPreview = data.photo;
      },
      error: (err) => {
        console.error('Erreur chargement profil:', err);
        alert('Erreur de chargement du profil');
      }
    });
  }

  onPhotoChange(event: any) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.photoPreview = e.target.result;
        this.profile.photo = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  }

  saveProfile() {
    if (!this.userId) return;

    // Préparer les données à envoyer (seulement les champs modifiés)
    const profileData: any = {};
    
    if (this.profile.fullName) profileData.full_name = this.profile.fullName;
    if (this.profile.phone) profileData.phone = this.profile.phone;
    if (this.profile.speciality) profileData.speciality = this.profile.speciality;
    if (this.profile.address) profileData.address = this.profile.address;
    if (this.profile.rpps) profileData.rpps = this.profile.rpps;
    if (this.profile.photo) profileData.photo = this.profile.photo;

    // Sauvegarder le profil
    this.authService.updateProfile(this.userId, profileData).subscribe({
      next: () => {
        // Gérer le changement de mot de passe si nécessaire
        if (this.newPassword) {
          if (!this.currentPassword) {
            alert('Veuillez saisir votre mot de passe actuel');
            return;
          }
          if (this.newPassword !== this.confirmPassword) {
            alert('Les nouveaux mots de passe ne correspondent pas');
            return;
          }
          if (this.newPassword.length < 6) {
            alert('Le mot de passe doit contenir au moins 6 caractères');
            return;
          }

          this.authService.changePassword(this.userId!, this.currentPassword, this.newPassword).subscribe({
            next: () => {
              alert('Profil et mot de passe mis à jour avec succès !');
              this.editMode = false;
              this.currentPassword = '';
              this.newPassword = '';
              this.confirmPassword = '';
              this.loadProfile();
            },
            error: (err) => {
              alert(err.error?.detail || 'Erreur lors du changement de mot de passe');
            }
          });
        } else {
          alert('Profil mis à jour avec succès !');
          this.editMode = false;
          this.loadProfile();
        }
      },
      error: (err) => {
        console.error('Erreur mise à jour:', err);
        alert('Erreur lors de la mise à jour du profil');
      }
    });
  }

  cancel() {
    this.loadProfile();
    this.editMode = false;
    this.currentPassword = '';
    this.newPassword = '';
    this.confirmPassword = '';
  }
}
