# CradleMetrics Version 3.0 - Future Roadmap
## "Predictive Intelligence & Early Screening"

Version 3.0 focuses on transitioning from diagnostic monitoring to **proactive screening** and **predictive analytics**. This roadmap details the three pillar features designed to maximize clinical impact and commercial value.

---

### 🤖 1. LLM-Powered "Clinical Narrative" Generator
Transition from raw data tables to synthesized clinical impressions using Natural Language Generation.
*   **Feature**: A "Smart Narrative" engine that writes professional clinical summaries based on extracted biometrics.
*   **Mechanism**: Integrating local LLMs (e.g., Llama 3) to convert metrics (percentiles, CPR ratios, AFI levels) into structured paragraphs.
*   **Example**: *"Patient shows asymmetric growth with AC at the 8th percentile; however, Doppler CPR remains normal at 1.4, suggesting compensated growth restriction. Recommend follow-up in 10 days."*
*   **Value**: Drastically reduces the time clinicians spend drafting the "Impression" section of reports.

### 📉 2. Predictive Outcome Modeling (Gaussian Forecasting)
Move from current status assessment to future risk prediction.
*   **Feature**: A "Growth Forecast" tool that projects fetal weight and biometric trajectories.
*   **Mechanism**: Implementing **Gaussian Process Regressors** (GPR) to analyze longitudinal history from the `ClinicalHistoryManager`.
*   **Value**: Proactively identifies patients on a trajectory toward **IUGR** or **Macrosomia** weeks before they cross the diagnostic threshold.

### 🧬 3. Genetic Marker Screening Support (11-13 Weeks)
Expand CradleMetrics into the critical first-trimester screening market.
*   **Feature**: Automated detection and measurement of early genetic indicators.
    *   **Nuchal Translucency (NT)**: Automated segmentation of the fluid-filled space at the back of the neck.
    *   **Nasal Bone (NB)**: Automated visibility check for the three-line sign.
*   **Target Conditions**: Early screening for Trisomy 21 (Down Syndrome), 18, and 13.
*   **Value**: Transforms the platform into an early-pregnancy risk assessment tool, increasing adoption in high-volume OB-GYN screening centers.

---

### 🛠️ Technical Roadmap & Assets

#### A. Dataset Integration
To "teach" the model to recognize mid-sagittal profiles and genetic markers, we will leverage:
*   **Primary Asset**: [Kaggle Dataset for Fetus Framework](https://www.kaggle.com/datasets/samanehgholami/dataset-for-fetus-framework) (1,500+ annotated sagittal images).
*   **Secondary Asset**: [PhysioNet FETAL_PLANES_DB](https://archive.physionet.org/content/fetal-ultrasound-scan-planes/1.0.0/) for diverse anatomical orientations.

#### B. Model Evolution
1.  **Class Expansion**: Update YOLOv8 detector to include a `fetal_profile` class.
2.  **Point Prompting**: Enhance the SAM pipeline to automatically "prompt" the NT region based on profile detection.
3.  **Calibrated Measurement**: Refine pixel-to-millimeter calibration specifically for the ultra-precise NT measurement range (1.0mm - 4.0mm).

---
**Status**: Planning Phase
**Target**: Version 3.0.0-Beta
