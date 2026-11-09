# Classifying Heartbeats from ECG Signals
### A Neural Network Built and Trained from Scratch in PyTorch
**Team Members:**
- **Rizwana Mubarak** 
- **Abdul Rehman**

---

## 1. Overview
This project implements a 3-layer fully connected neural network (`187 -> 128 -> 64 -> 5`) **from first mathematical principles in PyTorch** to classify single-heartbeat ECG waveforms from the MIT-BIH Arrhythmia Dataset into five AAMI categories (`N`, `S`, `V`, `F`, `Q`).

All core algorithmic components are derived and written manually without high-level library abstractions:
- Parameter tensors ($W^{[\ell]}, b^{[\ell]}$) initialized with **He (Kaiming) normal initialization**.
- Custom **ReLU** activation via tensor clamping (`torch.clamp(z, min=0)`).
- Numerically stable **Cross-Entropy Loss** using the log-sum-exp shift trick.
- Hand-computed **Analytical Output-Layer Gradient Check** verified against PyTorch autograd within $10^{-8}$.
- Custom **Mini-Batch SGD with Momentum ($\beta=0.9$) and $L_2$ Weight Decay ($\lambda=10^{-4}$)**.
- Custom evaluation metrics: Accuracy, $5 \times 5$ Confusion Matrix, Per-Class Accuracy, Precision, Recall, and Balanced Accuracy.

---

## 2. Repository Layout
```text
├── data/                               # Dataset files (mitbih_train.csv & mitbih_test.csv)
├── src/
│   ├── load_data.py                    # Raw CSV dataset loader
│   ├── plot_beats.py                   # Plots 1 example heartbeat per class
│   ├── prepare.py                      # 90/10 train-val split, tensor conversion
│   ├── model.py                        # ECGNet architecture & custom cross-entropy
│   ├── grad_check.py                   # Hand-computed output layer gradient check
│   ├── train.py                        # Custom SGD + momentum + weight decay & LR search
│   ├── metrics.py                      # Custom accuracy, confusion matrix, balanced accuracy
│   └── test_model.py                   # Test set evaluation and confusion matrix
├── results/
│   ├── example_beats.png               # Five characteristic ECG morphologies
│   ├── lr_comparison_val_loss.png      # Validation loss across learning rates
│   ├── loss_curves.png                 # Loss curves over 30 epochs
│   ├── accuracy_curves.png             # Accuracy curves over 30 epochs
│   ├── confusion_matrix.png            # Normalized confusion matrix heatmap
│   ├── tables.pdf                      # Times New Roman B&W formatted tables
│   ├── report_and_analysis.pdf         # Times New Roman B&W 4-page analysis report
│   ├── medium_blog.md                  # Comprehensive technical Medium article
│   ├── linkedin_post.md                # Professional LinkedIn announcement
│   ├── poster.html                     # Classic minimal academic poster (HTML/CSS)
│   ├── poster.pdf                      # Classic minimal academic poster (PDF)
│   └── poster.png                      # High-resolution minimal academic poster (PNG)
├── app/
│   ├── app.py                          # Task 7.1: Spatial UI Web Application (VisionOS HUD)
│   ├── curated_beats.json              # Canonical ECG beats across 5 classes
│   └── model_weights.json              # Client-side weights for real-time inference
├── ECG_Heartbeat_Classification.ipynb  # Self-contained Jupyter Notebook
└── README.md
```

---

## 3. How to Run Everything

### Step 1: Install Dependencies
```bash
pip install torch numpy pandas matplotlib streamlit reportlab
```

### Step 2: Place Dataset Files
Place `mitbih_train.csv` and `mitbih_test.csv` inside `data/` (or run directly in `AML_A3/` where datasets are already placed).

### Step 3: Run Tasks in Sequence
```bash
# 1. Verify raw data and plot example beats
python src/load_data.py
python src/plot_beats.py

# 2. Run data preparation and 90/10 split (Task 1)
python src/prepare.py

# 3. Model parameter count (Task 2) and analytical gradient check (Task 3)
python src/model.py
python src/grad_check.py

# 4. Train network with learning rate sweep and save best model (Task 4)
python src/train.py

# 5. Evaluate best model on unseen test set (Task 5)
python src/test_model.py
```

### Step 4: Launch the Spatial Telemetry Web Application (Task 7.1)
Run using the signed Python interpreter:
```bash
python -m streamlit run app/app.py
```

---

## 4. Key Empirical Results
- **Trainable Parameters:** `32,645`
- **Output-Layer Gradient Verification:** Max absolute difference `< 6.0e-08`
- **Overall Test Accuracy:** **`97.81%`** (21,412 / 21,892 correct)
- **Macro Balanced Accuracy:** **`82.53%`**
- **Test Set Loss:** `0.0827`
- **Per-Class Metrics:**
  - **N (Normal):** Precision `98.41%` | Recall `99.45%` | F1-Score `98.93%`
  - **S (Supraventricular):** Precision `78.43%` | Recall `61.15%` | F1-Score `68.72%`
  - **V (Ventricular Ectopic):** Precision `94.62%` | Recall `93.30%` | F1-Score `93.96%`
  - **F (Fusion):** Precision `68.22%` | Recall `54.32%` | F1-Score `60.48%`
  - **Q (Paced / Unknown):** Precision `98.98%` | Recall `97.45%` | F1-Score `98.21%`

---

## 5. VisionOS Spatial UI Design
The web interface was engineered following **Spatial UI principles** and the strict **60-30-10 color rule**:
* **60% Midnight Black (`#0B0B0C`):** Deep ambient spatial background with subtle glowing light cones.
* **30% Pure White (`#FFFFFF`):** Luminous typography, frosted glass card borders (`backdrop-filter: blur(28px)`), and crisp telemetry numbers.
* **10% Accent (Electric Cyan `#00F2FE` & Crimson `#FF3366`):** State-dependent indicators for normal rhythm versus arrhythmia warnings.
* **Interactive Features:**
  - **Tactile Beat Selector Pods:** Five cards with live mini SVG waveforms for N, S, V, F, Q.
  - **60 FPS Vector Oscilloscope:** Glowing neon ECG trace with an animated laser sweep scanline.
  - **Radial Confidence Speedometer:** Circular arc meter with gradient neon stroke.
  - **Interactive Noise & Wander Dials:** Real-time client-side signal distortion testing.
  - **Heartbeat Synthesizer:** Physiological lub-dub audio pulse using the Web Audio API.

---

## 6. Project Authors
* **Rizwana Mubarak** 
* **Abdul Rehman**  

