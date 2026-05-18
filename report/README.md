# Project Report - Hybrid Multi-Modal Phishing Detection System

This folder contains the LaTeX source files for the Major Project Report.

## Report Structure

The report follows the standard B.Tech Major Project Report format:

1. **Title Page**
2. **Supervisor's Certificate**
3. **Acknowledgement**
4. **Declaration of Originality**
5. **Abstract**
6. **Table of Contents**
7. **List of Figures**
8. **List of Tables**
9. **Chapter 1: Introduction** (Problem Statement, Objectives)
10. **Chapter 2: Literature Survey**
11. **Chapter 3: Methodology**
12. **Chapter 4: System Workflow**
13. **Chapter 5: Implementation**
14. **Chapter 6: Experiments and Results**
15. **Chapter 7: Conclusion and Future Scope**
16. **References**
17. **Appendix: Python Implementation Code**

## Compilation Instructions

### Using pdflatex (Recommended)

```bash
cd report
pdflatex main.tex
pdflatex main.tex   # Run twice for TOC and references
```

### Using latexmk

```bash
latexmk -pdf main.tex
```

### Using Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com/)
2. Upload `main.tex`
3. Click "Recompile" to generate PDF

### Required LaTeX Packages

The following packages are used in this report:
- inputenc, fontenc (encoding)
- graphicx (images)
- geometry (page layout)
- setspace (line spacing)
- titlesec (chapter formatting)
- fancyhdr (headers/footers)
- hyperref (hyperlinks)
- listings (code listings)
- xcolor (colors)
- amsmath, amssymb (math)
- booktabs, array, longtable (tables)
- caption, subcaption (figure captions)
- float (figure placement)
- algorithm, algorithmic (algorithms)
- tocloft (table of contents)

Most of these are included in standard LaTeX distributions (TeX Live, MiKTeX).

## Customization

### Adding Institute Logo

To add the MNIT logo on the title page:

1. Place your logo image (e.g., `mnit_logo.png`) in the `report/` folder
2. Uncomment this line in `main.tex`:
   ```latex
   % \includegraphics[width=0.2\textwidth]{mnit_logo.png}
   ```

### Updating Author Information

Edit the title page section in `main.tex` to update:
- Student name and ID
- Supervisor name
- Date
- Department information

### Adding Figures

To add figures:

```latex
\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{your_figure.png}
\caption{Your caption here}
\label{fig:your_label}
\end{figure}
```

## Output

After successful compilation, the report will be generated as `main.pdf`.

## Project Overview

The report documents a **Hybrid Multi-Modal Phishing Detection System** that combines:

- **URL Analysis Module**: Extracts 30+ features from URLs and uses ML models (XGBoost, Random Forest, LightGBM)
- **Visual Analysis Module**: Analyzes webpage screenshots using deep learning (ResNet, ViT, EfficientNet)
- **LLM Explainability Module**: Generates human-readable explanations using GPT-4 or local models

The system achieves **96.12% accuracy** on phishing detection, outperforming single-modality approaches.

## Author

Govind Ram Mali (2022UCP1630)

Department of Computer Science & Engineering  
Malaviya National Institute of Technology Jaipur

December, 2025
