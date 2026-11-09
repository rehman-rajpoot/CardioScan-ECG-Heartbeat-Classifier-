# Task 6: Analysis & Results

**Team Members:**
1. **Rizwana Mubarak** — Roll No: `23F-6055`
2. **Abdul Rehman** — Roll No: `23F-6075`

---

## Task 1.5: Morphological Differences Between the Five Beat Types
Normal beats (**N**) show a clean, narrow QRS peak followed by a smooth return to the baseline, while ventricular ectopic beats (**V**) originate directly in the ventricles and have a much wider, deeper, and notched waveform. Supraventricular beats (**S**) have a narrow peak similar to normal beats but fire prematurely, fusion beats (**F**) look like a mixture of normal and ventricular shapes, and unknown/paced beats (**Q**) display sharp artificial pacemaker spikes and unusual decay shapes.

---

## Task 6 Questions

### 6.1 Always-"N" Baseline vs. Our Model
- **Always-"N" Test Accuracy**:
  $$\frac{18,118}{21,892} = 82.76\%$$
- **Comparison**: Our trained neural network reaches **$97.81\%$** test accuracy (and **$82.53\%$** balanced accuracy), which is $15.05$ percentage points higher than the dummy baseline.
- **What this tells us about using accuracy alone**: Because Normal (**N**) beats make up $82.76\%$ of the test set, a useless classifier that labels every single patient as normal still gets $82.76\%$ accuracy while missing $100\%$ of abnormal arrhythmias. Therefore, overall accuracy cannot be trusted alone on imbalanced datasets; we must also check per-class accuracy and balanced accuracy.

### 6.2 Lowest Per-Class Accuracy & Most Common Mistake
- **Lowest Per-Class Accuracy**: **Class F (Fusion beat)** has the lowest per-class accuracy at **$54.32\%$** ($88 / 162$), followed by **Class S** at **$69.24\%$** ($385 / 556$).
- **Most Often Mistaken For**: In the confusion matrix, out of $162$ true Class F beats, **$54$ are mistaken for Class N (Normal)** and **$19$ are mistaken for Class V (Ventricular)**.
- **Why they are hard to separate**: Physiologically, a fusion beat happens when a normal sinus impulse and a ventricular ectopic impulse activate the heart ventricles at the exact same time. Because of this, its ECG waveform is literally a blend of a Normal (N) beat and a Ventricular (V) beat. Additionally, Class F has only $641$ training rows ($0.73\%$ of the data), giving the network very few examples to learn its boundary.

### 6.3 Loss Curves & Overfitting Analysis
- Looking at the loss curves for $\alpha = 0.5$, both training loss and validation loss drop quickly in the first 10 epochs and flatten out smoothly up to epoch 30.
- At the best epoch (Epoch 25), the training loss is **$0.0535$** and the validation loss is **$0.0716$** (a small gap of only $\sim 0.018$).
- Because both losses are low and the validation loss does not curve upward as training continues, the network is **neither underfitting nor overfitting**. The $L_2$ weight decay ($\lambda = 10^{-4}$) and large training set ($78,799$ rows) help the network generalize well.

### 6.4 Why Not Zero Initialization & Why the He Rule Uses $\sqrt{2 / n^{[\ell-1]}}$
- **Why not initialize every weight to zero**: If all weights $W^{[\ell]} = 0$, every hidden unit in a layer receives the exact same input sum ($0$), produces the exact same output, and receives the exact same gradient during backpropagation. Every neuron in the layer would update identically and learn the exact same feature (the **symmetry problem**), making the hidden layer no better than a single neuron.
- **Why He initialization uses $\sqrt{2 / n^{[\ell-1]}}$ for ReLU layers**: Standard variance scaling uses $1 / n^{[\ell-1]}$ to keep signal variance constant from one layer to the next. However, $\text{ReLU}(z) = \max(z, 0)$ sets all negative inputs to zero, which cuts the variance of activations in half at each hidden layer. Multiplying the weight variance by $2$ (giving standard deviation $\sqrt{2 / n^{[\ell-1]}}$) compensates for the half lost to ReLU so signals and gradients neither vanish nor explode across layers.

### 6.5 Momentum on Opposite Gradients & Weight Decay with Zero Gradient
- **Momentum when consecutive gradients point in opposite directions**: Since velocity is updated as $v_P = \beta v_P + (1 - \beta) g$, opposite signs in consecutive mini-batch gradients partially cancel each other out in $v_P$. This dampens zig-zagging oscillations across steep directions while keeping steady progress along consistent directions.
- **Weight decay on a weight with zero gradient**: When $\frac{\partial J}{\partial P} = 0$, the gradient term becomes purely $g = \lambda P$. The update rule shrinks the weight towards zero at each step ($P \leftarrow P - \alpha v_P$), preventing unused weights from staying unnecessarily large.

### 6.6 One Change to Help Rare Classes (Class F)
- **Suggested Change**: Use **Class-Weighted Cross-Entropy Loss**.
- **Explanation**: Instead of treating all rows equally in the loss, we multiply the loss of class $c$ by an inverse frequency weight $w_c = \frac{N_{\text{total}}}{5 \times N_c}$, so the loss becomes $L = -w_y \log \text{softmax}(z)_y$. Since Class F has only $641$ training rows compared to $72,471$ for Class N, misclassifying a rare Class F beat will produce a much larger gradient penalty than misclassifying a normal beat, forcing the network to improve recall on rare classes.
