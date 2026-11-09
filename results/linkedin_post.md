# 🫀 Building an ECG Heartbeat Classifier from Scratch in PyTorch (Zero High-Level Abstractions)

Proud to share our latest Machine Learning project: classifying cardiac arrhythmias from electrocardiogram (ECG) signals using a custom 3-layer neural network built **entirely from mathematical first principles in PyTorch**!

Together with my teammate **Rizwana Mubarak (23F-6055)**, we challenged ourselves to bypass all high-level frameworks:
🚫 No `torch.nn.Linear` or `torch.nn.Sequential`  
🚫 No `torch.optim` (no built-in Adam/SGD)  
🚫 No `torch.nn.CrossEntropyLoss`  
🚫 No `scikit-learn` for metrics or evaluation  

Instead, every single component was derived and coded from scratch:
✨ **He (Kaiming) Normal Weight Initialization** ($\sigma = \sqrt{2/n_{in}}$)  
✨ **Custom Forward Pass** with manual clamp ReLU activations  
✨ **Numerically Stable Log-Sum-Exp Cross Entropy**  
✨ **Manual SGD with Momentum ($\beta=0.9$) and L2 Weight Decay**  
✨ **Class Imbalance Mitigation** addressing the 113:1 Normal-to-Fusion ratio  

### 📊 Benchmark Results on MIT-BIH Arrhythmia Dataset:
* 🎯 **Test Accuracy:** **97.81%** across 21,892 unseen heartbeats  
* ⚖️ **Balanced Macro Accuracy:** **82.53%**  
* ⚡ **Inference Latency:** **< 0.3 ms** per 187-sample cardiac vector  
* 📈 **Normal Sinus (N) F1:** **98.93%** | **Paced (Q) F1:** **98.21%** | **Ventricular (V) F1:** **93.96%**  

### 🌌 VisionOS Spatial Telemetry Web UI:
To make model testing interactive and engaging, we engineered a sleek **Spatial Telemetry HUD** built around the 60-30-10 color rule (Midnight Black `#0B0B0C`, Pure White `#FFFFFF`, and Electric Cyan `#00F2FE` / Crimson `#FF3366`):
* 60 FPS real-time neon vector oscilloscope with laser scanlines
* 5 holographic tactile beat pods with embedded SVG waveform previews
* Circular radial confidence arc gauge
* Interactive live signal noise and respiratory wander injection dials
* Physiological heartbeat audio synthesizer

Check out our full technical writeup, mathematical derivations, and open-source code:
📖 **Medium Technical Article:** [Link in comments]  
💻 **GitHub Repository:** [https://github.com/your-username/ECG_Heartbeat_Classification]  
📊 **Kaggle Notebook:** [Link in comments]  

Huge appreciation to our instructor and peers for guidance throughout this course!

**Team:**  
👤 **Rizwana Mubarak** (`23F-6055`)  
👤 **Abdul Rehman** (`23F-6075`)  

#MachineLearning #DeepLearning #PyTorch #HealthTech #Cardiology #ECG #AIinHealthcare #FromScratch #DataScience #ArtificialIntelligence
