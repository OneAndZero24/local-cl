# Statistical Analysis of Local Methods vs ReLU Across Multiple Datasets
---

## Goal

For each dataset \( d \), test:

\[
H_0^{(d,m)}: \quad \mu_{m,d} = \mu_{\text{ReLU},d}
\]

vs

\[
H_1^{(d,m)}: \quad \mu_{m,d} \neq \mu_{\text{ReLU},d}
\]

for each local method 

\[
m \in \{ \text{Local LT}, \text{Local LT Head}, \text{Local}, \text{Local Head} \}.
\]

Here, \( \mu_{m,d} \) denotes the true mean performance of method \( m \) on dataset \( d \).

---

## Step 1: Paired Tests Per Dataset

### Data

For dataset \( d \), paired observations:

\[
\{ (X_{i}^{(\text{ReLU}, d)}, X_{i}^{(m,d)}) \}_{i=1}^{n_d}
\]

where \( n_d \) is the number of paired observations.

---

### Paired t-test

Evaluate whether the mean difference

\[
D_i^{(m,d)} = X_{i}^{(m,d)} - X_{i}^{(\text{ReLU}, d)}
\]

has zero mean, i.e.,

\[
H_0^{(d,m)}: \quad \mu_{D}^{(m,d)} = 0 \quad \text{vs} \quad H_1^{(d,m)}: \quad \mu_{D}^{(m,d)} \neq 0
\]

Test statistic:

\[
t^{(d,m)} = \frac{\bar{D}^{(m,d)}}{s_D^{(m,d)} / \sqrt{n_d}}
\]

where

- \( \bar{D}^{(m,d)} = \frac{1}{n_d} \sum_{i=1}^{n_d} D_i^{(m,d)} \) (sample mean difference),
- \( s_D^{(m,d)} = \sqrt{\frac{1}{n_d - 1} \sum_{i=1}^{n_d} (D_i^{(m,d)} - \bar{D}^{(m,d)})^2} \) (sample std dev of differences).

Under \( H_0 \), \( t^{(d,m)} \) follows a Student’s t-distribution with \( n_d - 1 \) degrees of freedom.

Calculate p-value \( p^{(d,m)} \) from this distribution.

---

## Step 2: Combine p-values Across Datasets for Each Method

For each method \( m \), you have p-values:

\[
\{ p^{(d,m)} : d \in \text{datasets} \}
\]

---

### Fisher’s Combined Probability Test

Aggregate these p-values into a single test:

\[
\chi^2 = -2 \sum_{d} \ln(p^{(d,m)})
\]

Under the null hypothesis, this statistic follows a chi-square distribution with \( 2k \) degrees of freedom, where \( k \) is the number of datasets.

Overall p-value:

\[
p_{\text{combined}}^{(m)} = P\left(\chi^2_{2k} \geq -2 \sum_{d} \ln(p^{(d,m)}) \right)
\]

---

## Step 3: Multiple Comparisons Correction

Testing multiple local methods \( m = 1, \ldots, M \) inflates Type I error rate.

---

### Bonferroni Correction

Control family-wise error rate (FWER) at level \( \alpha \) (e.g., 0.05):

\[
\alpha' = \frac{\alpha}{M}
\]

Method \( m \) is significant only if:

\[
p_{\text{combined}}^{(m)} < \alpha'
\]

---

## Step 4: Effect Direction and Magnitude

Calculate average difference in means across datasets:

\[
\bar{\Delta}^{(m)} = \frac{1}{k} \sum_{d} \bar{D}^{(m,d)} = \frac{1}{k} \sum_{d} \left( \frac{1}{n_d} \sum_{i=1}^{n_d} \big(X_{i}^{(m,d)} - X_{i}^{(\text{ReLU}, d)}\big) \right)
\]

- \( \bar{\Delta}^{(m)} > 0 \) means method \( m \) tends to outperform ReLU.
- \( \bar{\Delta}^{(m)} < 0 \) means it tends to perform worse.

---

## Interpretation Summary

- If 

\[
p_{\text{combined}}^{(m)} < \alpha' \quad \text{and} \quad \bar{\Delta}^{(m)} > 0,
\]

then method \( m \) significantly improves over ReLU.

- If 

\[
p_{\text{combined}}^{(m)} < \alpha' \quad \text{but} \quad \bar{\Delta}^{(m)} < 0,
\]

then method \( m \) differs significantly but is worse on average.

- Otherwise, no significant improvement is detected.

---