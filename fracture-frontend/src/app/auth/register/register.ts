import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule, Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './register.html',
  styleUrls: ['./register.scss']
})
export class RegisterComponent {
  fullName = '';
  email = '';
  phone = '';
  speciality = '';
  address = '';
  rpps = '';
  password = '';
  confirmPassword = '';
  acceptTerms = false;
  showPassword = false;
  showConfirmPassword = false;
  loading = false;
  error = '';

  constructor(
    private router: Router,
    private authService: AuthService
  ) {}

  onSubmit() {
    if (this.password !== this.confirmPassword) {
      alert('Les mots de passe ne correspondent pas');
      return;
    }

    if (!this.acceptTerms) {
      alert('Veuillez accepter les conditions d\'utilisation');
      return;
    }

    this.loading = true;
    this.error = '';

    // Appeler l'API avec toutes les infos
    const registerData = {
      full_name: this.fullName,
      email: this.email,
      password: this.password,
      phone: this.phone,
      speciality: this.speciality,
      address: this.address,
      rpps: this.rpps
    };

    this.authService.register(
      this.fullName,
      this.email,
      this.password,
      this.phone,
      this.speciality,
      this.address,
      this.rpps
    ).subscribe({
      next: (response) => {
        console.log('Register success:', response);
        // Sauvegarder l'ID utilisateur
        localStorage.setItem('userId', response.user.id.toString());
        alert(`Compte créé avec succès ! Bienvenue ${response.user.first_name}`);
        this.router.navigate(['/auth/login']);
      },
      error: (err) => {
        this.loading = false;
        this.error = err.error?.detail || 'Erreur d\'inscription';
        alert(this.error);
      },
      complete: () => {
        this.loading = false;
      }
    });
  }
}
