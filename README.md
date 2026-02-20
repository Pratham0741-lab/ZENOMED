# Zenomed - Pharmacogenomic Risk Intelligence Platform

![Zenomed Logo](logo.png)

## 🧬 Overview

Zenomed is an advanced pharmacogenomic decision-support tool that analyzes genetic variants from VCF files to provide clinical-grade drug safety insights. Built for the RIFT 2026 Hackathon, it combines CPIC guidelines, DDGI phenoconversion engines, and Google Gemini 2.5 Flash AI to deliver personalized medication recommendations.

## ✨ Features

- **VCF File Analysis** - Native parser supporting standard VCF format (max 10MB)
- **CPIC Engine** - Evidence-graded clinical recommendations based on PharmGKB guidelines
- **DDGI Phenoconversion** - Dynamic drug-drug-gene interaction analysis
- **AI-Powered Insights** - Google Gemini 2.5 Flash integration for explanations
- **6 Target Genes** - CYP2D6, CYP2C19, CYP2C9, VKORC1, SLCO1B1, CYP3A5
- **21 Drug Pairs** - Comprehensive medication coverage
- **Real-time Analytics** - KMeans clustering and interactive visualizations
- **Concomitant Medication Tracking** - CYP inhibitor interaction analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Google Gemini API key (optional, for AI features)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd zenomed
```

2. **Install Python dependencies**
```bash
pip install flask flask-cors pandas numpy scikit-learn google-generativeai
```

3. **Start the backend server**
```bash
python backend.py
```
The server will start on `http://localhost:5000`

4. **Open the frontend**
```bash
# Simply open index.html in your browser
open index.html
# Or use a local server
python -m http.server 8000
```

## 📁 Project Structure

```
zenomed/
├── index.html              # Main frontend application
├── backend.py              # Flask API server
├── logo.png               # Zenomed logo
├── README.md              # This file
├── generated_vcfs/        # Sample VCF files for testing
│   ├── case1_clopidogrel_poor.vcf
│   ├── case2_clopidogrel_intermediate.vcf
│   ├── case3_warfarin_sensitive.vcf
│   └── ... (10 test cases)
└── invalid_vcfs/          # Invalid VCF files for validation testing
    ├── invalid_header_syntax.vcf
    ├── invalid_info_format.vcf
    └── ...
```

## 🎯 Usage

### 1. Upload VCF File
- Click "Click to Upload VCF" button
- Select a valid VCF file (or use demo data)
- Maximum file size: 10MB

### 2. Enter Gemini API Key (Optional)
- Obtain API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Paste into the "Gemini API Key" field
- Leave empty to use demo mode

### 3. Select Concomitant Medications
- Click the dropdown to select any concurrent medications
- Includes CYP inhibitors (Bupropion, Fluoxetine, Paroxetine, etc.)
- System will analyze drug-drug-gene interactions

### 4. Enter Patient ID
- Default: PATIENT-001
- Customize for your records

### 5. Analyze Genome
- Click "ANALYZE GENOME →" button
- Processing overlay will show analysis steps
- Results appear in interactive cards

## 📊 Results Dashboard

### Card 1: Clinical Pharmacogenomic Report
- Evidence-graded recommendations
- Gene/variant analysis table
- Risk percentages (0-100%)
- Downloadable JSON report

### Card 2: Risk Predictor
- Visual risk score (0-100)
- Color-coded severity levels
- Animated risk visualization

### Card 3: AI Explanations
- Gemini-powered insights
- Layman-friendly explanations
- Clinical context

### Card 4: Analytics Dashboard
- 2x2 grid of interactive charts
- Risk distribution
- Gene frequency analysis
- Drug category breakdown
- Phenotype clustering

## 🔧 API Endpoints

### POST `/api/analyze`
Analyzes VCF file and returns pharmacogenomic report.

**Request (multipart/form-data):**
```
patient_id: string
gemini_key: string (optional)
concomitant_meds: string (comma-separated)
vcf_file: file (optional)
```

**Response:**
```json
{
  "patient_id": "PATIENT-001",
  "genes_analyzed": 6,
  "drug_pairs_evaluated": 21,
  "vcf_parsing_integrity": "100%",
  "recommendations": [...],
  "risk_score": 45,
  "ai_explanation": "...",
  "analytics": {...}
}
```

## 🧪 Testing

### Sample VCF Files
Use the provided test cases in `generated_vcfs/`:
- `case1_clopidogrel_poor.vcf` - Poor metabolizer
- `case3_warfarin_sensitive.vcf` - Warfarin sensitivity
- `case5_codeine_toxic.vcf` - Toxicity risk
- `case7_statin_myopathy.vcf` - Myopathy risk

### Invalid VCF Testing
Test error handling with files in `invalid_vcfs/`:
- Missing headers
- Invalid format
- Syntax errors

## ⚡ Performance Optimizations

- **Hardware Acceleration** - GPU-accelerated animations (60 FPS)
- **CSS Containment** - Isolated rendering for better performance
- **Deferred Loading** - Non-blocking script execution
- **Resource Preloading** - Critical fonts and assets
- **Transform Optimization** - `translateZ(0)` for smooth scrolling

## 🎨 Design Features

- **Apple San Francisco Font** - Native system fonts for optimal readability
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Dark Theme** - Eye-friendly navy blue color scheme
- **Smooth Animations** - GSAP-powered scroll effects
- **Interactive Charts** - Chart.js visualizations
- **Custom Cursor** - Enhanced user experience

## 🔒 Security & Compliance

⚠️ **Medical Disclaimer**: Zenomed is an investigational tool intended exclusively for licensed healthcare professionals. All outputs are for research and educational purposes only and should not replace professional medical advice.

- Patient data is processed locally
- No data stored on servers
- API keys handled securely
- HIPAA considerations for production use

## 🛠️ Technology Stack

**Frontend:**
- HTML5, CSS3, JavaScript (ES6+)
- GSAP 3.12.5 (animations)
- Chart.js 4.4.0 (visualizations)
- Apple San Francisco fonts

**Backend:**
- Python 3.8+
- Flask (web framework)
- Flask-CORS (cross-origin support)
- Pandas (data processing)
- NumPy (numerical computing)
- Scikit-learn (ML analytics)
- Google Generative AI (Gemini integration)

## 📝 Configuration

### Backend Configuration
Edit `backend.py` to customize:
- Port number (default: 5000)
- CORS settings
- File upload limits
- API endpoints

### Frontend Configuration
Edit `index.html` to customize:
- API endpoint URL
- Color scheme (CSS variables)
- Animation timings
- Chart configurations

## 🐛 Troubleshooting

### Processing Screen Not Showing
- Check browser console (F12) for errors
- Ensure backend is running on port 5000
- Verify CORS is enabled

### VCF Upload Fails
- Check file size (max 10MB)
- Verify VCF format is valid
- Ensure backend server is running

### Charts Not Rendering
- Verify Chart.js is loaded
- Check browser console for errors
- Ensure data format is correct

## 🤝 Contributing

This project was created for the RIFT 2026 Hackathon by Shadow Syndicate.

**Team:**
- Lead Bioinformatics Software Engineer & UI/UX Design

## 📄 License

© 2025-2026 Zenomed · RIFT Hackathon

## 🔗 Resources

- [CPIC Guidelines](https://cpicpgx.org/)
- [PharmGKB Database](https://www.pharmgkb.org/)
- [Google Gemini API](https://ai.google.dev/)
- [VCF Format Specification](https://samtools.github.io/hts-specs/VCFv4.2.pdf)

## 🌐 Links

- [LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:ugcPost:7430429417744752640/)
- [GitHub Repository](https://github.com/Pratham0741-lab/ZENOMED)
- [Live Deployed Web Application](https://cozy-beijinho-4221e4.netlify.app/)

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ for RIFT 2026 Hackathon**
