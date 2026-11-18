import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { SidebarComponent } from '../shared/sidebar/sidebar';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.scss']
})
export class DashboardComponent {
  selectedImage: string | null = null;
  selectedFile: File | null = null;
  imageSize: number = 100;
  analyzing: boolean = false;
  analysisResult: any = null;
  isFullscreen: boolean = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {}

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.selectedImage = e.target.result;
        this.cdr.markForCheck();
      };
      reader.readAsDataURL(file);
    }
  }

  analyzeImage() {
    if (!this.selectedFile) return;

    this.analyzing = true;
    this.cdr.markForCheck();
    
    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post('http://localhost:8000/api/analyze', formData).subscribe({
      next: (response: any) => {
        this.analysisResult = {
          diagnosis: response.diagnosis,
          confidence: response.confidence,
          fractured: response.fractured,
          efficientnet: response.efficientnet,
          yolo: response.yolo,
          annotatedImage: response.images.annotated,
          agreement: response.agreement
        };
        this.analyzing = false;
        this.cdr.markForCheck();
      },
      error: (err) => {
        console.error('Error analyzing image:', err);
        alert('Erreur lors de l\'analyse');
        this.analyzing = false;
        this.cdr.markForCheck();
      }
    });
  }

  resetAnalysis() {
    this.selectedImage = null;
    this.selectedFile = null;
    this.analysisResult = null;
    this.imageSize = 100;
    this.isFullscreen = false;
    this.cdr.markForCheck();
  }

  toggleFullscreen() {
    this.isFullscreen = !this.isFullscreen;
    this.cdr.markForCheck();
  }
}
