import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

interface AuthResponse {
  token: string;
  user: {
    id: number;
    email: string;
    first_name: string;
    last_name: string;
  };
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://localhost:8000/api/auth';
  private tokenKey = 'fracture_token';

  constructor(private http: HttpClient) {}

  register(
    fullName: string,
    email: string,
    password: string,
    phone?: string,
    speciality?: string,
    address?: string,
    rpps?: string
  ): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/register`, {
      full_name: fullName,
      email,
      password,
      phone,
      speciality,
      address,
      rpps
    }).pipe(
      tap(response => {
        this.saveToken(response.token);
        localStorage.setItem('userId', response.user.id.toString());
      })
    );
  }

  login(email: string, password: string): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/login`, {
      email,
      password
    }).pipe(
      tap(response => {
        this.saveToken(response.token);
        localStorage.setItem('userId', response.user.id.toString());
      })
    );
  }

  getProfile(userId: number): Observable<any> {
    return this.http.get(`http://localhost:8000/api/profile/${userId}`);
  }

  updateProfile(userId: number, profile: any): Observable<any> {
    return this.http.put(`http://localhost:8000/api/profile/${userId}`, profile);
  }

  changePassword(userId: number, currentPassword: string, newPassword: string): Observable<any> {
    return this.http.put(`http://localhost:8000/api/profile/${userId}/password`, {
      current_password: currentPassword,
      new_password: newPassword
    });
  }

  private saveToken(token: string): void {
    localStorage.setItem(this.tokenKey, token);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }
}
