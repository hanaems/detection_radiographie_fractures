import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: '/auth/login', pathMatch: 'full' },
  { 
    path: 'auth/login', 
    loadComponent: () => import('./auth/login/login').then(m => m.LoginComponent)
  },
  { 
    path: 'auth/register', 
    loadComponent: () => import('./auth/register/register').then(m => m.RegisterComponent)
  },
  { 
    path: 'profile', 
    loadComponent: () => import('./profile/profile').then(m => m.ProfileComponent)
  },
  { 
    path: 'dashboard',
    loadComponent: () => import('./dashboard/dashboard').then(m => m.DashboardComponent)
  }
];
