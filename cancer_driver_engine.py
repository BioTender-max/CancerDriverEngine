import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Simulate 500 genes × 200 tumors ─────────────────────────────────────────
N_GENES = 500
N_TUMORS = 200
N_DRIVERS = 30

gene_names = [f'GENE{i:04d}' for i in range(N_GENES)]
# Assign some real-sounding driver names
real_drivers = ['TP53', 'KRAS', 'PIK3CA', 'PTEN', 'APC', 'EGFR', 'BRAF', 'MYC',
                'RB1', 'CDKN2A', 'VHL', 'BRCA1', 'BRCA2', 'IDH1', 'ARID1A',
                'CTNNB1', 'SMAD4', 'STK11', 'NF1', 'RET', 'ALK', 'MET', 'FGFR1',
                'CDH1', 'FBXW7', 'NOTCH1', 'PTCH1', 'SMARCA4', 'KMT2D', 'DNMT3A']
for i, name in enumerate(real_drivers):
    gene_names[i] = name

# Mutation matrix (binary: mutated or not)
mutation_matrix = np.random.binomial(1, 0.02, size=(N_GENES, N_TUMORS))
# Driver genes have higher mutation rates
driver_indices = list(range(N_DRIVERS))
driver_rates = np.random.uniform(0.05, 0.60, N_DRIVERS)
for i, rate in zip(driver_indices, driver_rates):
    mutation_matrix[i] = np.random.binomial(1, rate, N_TUMORS)

# dN/dS calculation
# Expected mutation rate under neutral model
expected_rate = 0.02
observed_rate = mutation_matrix.mean(axis=1)
# dN/dS = observed_nonsynonymous / expected_nonsynonymous
# Simplified: dN/dS ~ observed_rate / expected_rate with noise
dnds = (observed_rate / expected_rate) * np.random.lognormal(0, 0.3, N_GENES)
dnds = np.clip(dnds, 0.01, 50)
# Drivers have elevated dN/dS
dnds[driver_indices] = np.random.uniform(2.0, 15.0, N_DRIVERS)

# Hotspot detection: recurrence > background
mutation_recurrence = mutation_matrix.sum(axis=1)
background_mean = mutation_recurrence[N_DRIVERS:].mean()
background_std = mutation_recurrence[N_DRIVERS:].std()
hotspot_z = (mutation_recurrence - background_mean) / (background_std + 1e-10)
is_hotspot = hotspot_z > 2.0

# Functional impact scoring (SIFT/PolyPhen-style: 0=benign, 1=damaging)
sift_scores = np.random.beta(2, 5, N_GENES)  # lower = more damaging
polyphen_scores = np.random.beta(3, 4, N_GENES)  # higher = more damaging
# Drivers have more damaging scores
sift_scores[driver_indices] = np.random.beta(8, 2, N_DRIVERS)  # high = damaging (inverted)
polyphen_scores[driver_indices] = np.random.beta(7, 2, N_DRIVERS)
combined_impact = (sift_scores + polyphen_scores) / 2

# Oncogene vs TSG classification
# Oncogenes: gain-of-function (hotspot mutations, specific positions)
# TSGs: loss-of-function (truncating mutations, spread across gene)
is_oncogene = np.zeros(N_GENES, dtype=bool)
is_tsg = np.zeros(N_GENES, dtype=bool)
for i in driver_indices:
    if np.random.random() < 0.4:
        is_oncogene[i] = True
    else:
        is_tsg[i] = True

# Pathway enrichment
pathways = {
    'Cell Cycle': ['TP53', 'RB1', 'CDKN2A', 'CCND1', 'CDK4'],
    'PI3K/AKT': ['PIK3CA', 'PTEN', 'AKT1', 'MTOR'],
    'RAS/MAPK': ['KRAS', 'BRAF', 'NF1', 'EGFR'],
    'WNT': ['APC', 'CTNNB1', 'AXIN1'],
    'DNA Repair': ['BRCA1', 'BRCA2', 'ATM', 'CHEK2'],
    'Chromatin': ['ARID1A', 'SMARCA4', 'KMT2D', 'DNMT3A'],
    'TGF-β': ['SMAD4', 'TGFBR2'],
    'Hippo': ['NF2', 'LATS1', 'YAP1'],
}
pathway_enrichment = {}
for pw, genes in pathways.items():
    n_driver_in_pw = sum(1 for g in genes if g in real_drivers[:N_DRIVERS])
    enrichment = n_driver_in_pw / len(genes) / (N_DRIVERS / N_GENES)
    pathway_enrichment[pw] = enrichment

# Cancer types
cancer_types = ['LUAD', 'SKCM', 'COAD', 'BRCA', 'BLCA', 'HNSC', 'STAD', 'GBM']
sample_cancer = np.random.choice(cancer_types, N_TUMORS)

# Cancer-type specificity of driver genes (top 10 drivers)
ct_specificity = np.zeros((10, len(cancer_types)))
for i in range(10):
    for j, ct in enumerate(cancer_types):
        ct_mask = sample_cancer == ct
        ct_specificity[i, j] = mutation_matrix[i, ct_mask].mean()

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Cancer Driver Gene Analysis Dashboard', fontsize=18,
             color='white', fontweight='bold', y=0.98)

DARK = '#161b22'
TEXT = 'white'
ACCENT = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# 1. dN/dS distribution
ax = axes[0, 0]
style_ax(ax, '1. dN/dS Ratio Distribution')
ax.hist(np.log10(dnds[N_DRIVERS:] + 0.01), bins=40, color=ACCENT, alpha=0.7, label='Passengers', density=True)
ax.hist(np.log10(dnds[driver_indices] + 0.01), bins=15, color=ACCENT2, alpha=0.8, label='Drivers', density=True)
ax.axvline(np.log10(1.0), color='yellow', lw=2, ls='--', label='dN/dS=1 (neutral)')
ax.set_xlabel('log10(dN/dS)', color=TEXT, fontsize=9)
ax.set_ylabel('Density', color=TEXT, fontsize=9)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 2. Hotspot mutation map (top 20 genes)
ax = axes[0, 1]
style_ax(ax, '2. Hotspot Mutation Map (Top 20 Genes)')
top20_idx = np.argsort(mutation_recurrence)[::-1][:20]
top20_names = [gene_names[i] for i in top20_idx]
top20_rec = mutation_recurrence[top20_idx]
top20_hotspot = is_hotspot[top20_idx]
colors_hs = [ACCENT2 if h else ACCENT for h in top20_hotspot]
bars = ax.barh(range(20), top20_rec, color=colors_hs, alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_names, fontsize=7, color=TEXT)
ax.set_xlabel('Mutation Recurrence (# tumors)', color=TEXT, fontsize=9)
patches = [mpatches.Patch(color=ACCENT2, label='Hotspot'), mpatches.Patch(color=ACCENT, label='Non-hotspot')]
ax.legend(handles=patches, fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 3. Functional impact scores
ax = axes[0, 2]
style_ax(ax, '3. Functional Impact Scores')
ax.scatter(sift_scores[N_DRIVERS:], polyphen_scores[N_DRIVERS:], c=ACCENT, alpha=0.3, s=10, label='Passengers')
ax.scatter(sift_scores[driver_indices], polyphen_scores[driver_indices], c=ACCENT2, alpha=0.8, s=40, label='Drivers', zorder=5)
ax.set_xlabel('SIFT Score (damaging→high)', color=TEXT, fontsize=9)
ax.set_ylabel('PolyPhen Score (damaging→high)', color=TEXT, fontsize=9)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 4. Oncogene vs TSG classification
ax = axes[1, 0]
style_ax(ax, '4. Oncogene vs TSG Classification')
n_onco = is_oncogene[driver_indices].sum()
n_tsg = is_tsg[driver_indices].sum()
n_other = N_DRIVERS - n_onco - n_tsg
categories = ['Oncogene', 'TSG', 'Other']
counts = [n_onco, n_tsg, n_other]
colors_cls = [ACCENT2, ACCENT, '#d2a8ff']
bars = ax.bar(categories, counts, color=colors_cls, edgecolor='#0d1117', alpha=0.85)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(count), ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Driver Genes', color=TEXT, fontsize=9)
ax.set_ylim(0, max(counts) * 1.2)

# 5. Driver gene network (simplified scatter with connections)
ax = axes[1, 1]
style_ax(ax, '5. Driver Gene Co-mutation Network')
# Co-mutation frequency between top 10 drivers
top10_idx = driver_indices[:10]
comut_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if i != j:
            comut_matrix[i, j] = (mutation_matrix[top10_idx[i]] & mutation_matrix[top10_idx[j]]).mean()
# Plot as network
angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
x_pos = np.cos(angles)
y_pos = np.sin(angles)
for i in range(10):
    for j in range(i+1, 10):
        if comut_matrix[i, j] > 0.05:
            lw = comut_matrix[i, j] * 10
            ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                    color=ACCENT, alpha=0.5, lw=lw)
ax.scatter(x_pos, y_pos, s=200, c=ACCENT2, zorder=5, edgecolors='white', lw=1)
for i, (x, y) in enumerate(zip(x_pos, y_pos)):
    ax.text(x * 1.2, y * 1.2, real_drivers[i], ha='center', va='center',
            color=TEXT, fontsize=7, fontweight='bold')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')
ax.set_facecolor(DARK)

# 6. Mutation recurrence (top 15 drivers)
ax = axes[1, 2]
style_ax(ax, '6. Driver Gene Mutation Recurrence')
top15_driver_idx = np.argsort(mutation_recurrence[driver_indices])[::-1][:15]
top15_names = [real_drivers[i] for i in top15_driver_idx]
top15_rec = mutation_recurrence[np.array(driver_indices)[top15_driver_idx]]
top15_pct = top15_rec / N_TUMORS * 100
colors_rec = [ACCENT2 if is_oncogene[driver_indices[i]] else ACCENT for i in top15_driver_idx]
ax.barh(range(15), top15_pct, color=colors_rec, alpha=0.85)
ax.set_yticks(range(15))
ax.set_yticklabels(top15_names, fontsize=8, color=TEXT)
ax.set_xlabel('Mutation Frequency (%)', color=TEXT, fontsize=9)
patches = [mpatches.Patch(color=ACCENT2, label='Oncogene'), mpatches.Patch(color=ACCENT, label='TSG')]
ax.legend(handles=patches, fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 7. Pathway enrichment
ax = axes[2, 0]
style_ax(ax, '7. Pathway Enrichment of Driver Genes')
pw_names = list(pathway_enrichment.keys())
pw_scores = list(pathway_enrichment.values())
sorted_idx = np.argsort(pw_scores)[::-1]
colors_pw = [ACCENT2 if pw_scores[i] > 2 else ACCENT for i in sorted_idx]
ax.barh(range(len(pw_names)), [pw_scores[i] for i in sorted_idx],
        color=colors_pw, alpha=0.85)
ax.set_yticks(range(len(pw_names)))
ax.set_yticklabels([pw_names[i] for i in sorted_idx], fontsize=8, color=TEXT)
ax.set_xlabel('Enrichment Score', color=TEXT, fontsize=9)
ax.axvline(1.0, color='yellow', lw=1.5, ls='--', label='Expected (null)')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 8. Cancer type specificity heatmap
ax = axes[2, 1]
style_ax(ax, '8. Driver Gene Cancer-Type Specificity')
im = ax.imshow(ct_specificity, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(cancer_types)))
ax.set_xticklabels(cancer_types, color=TEXT, fontsize=8, rotation=45)
ax.set_yticks(range(10))
ax.set_yticklabels(real_drivers[:10], color=TEXT, fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Mutation Freq').ax.tick_params(colors=TEXT, labelsize=7)

# 9. Summary
ax = axes[2, 2]
style_ax(ax, '9. Analysis Summary')
ax.axis('off')
n_sig_drivers = (dnds > 2.0).sum()
summary_lines = [
    ('Total Genes', f'{N_GENES}'),
    ('Tumor Samples', f'{N_TUMORS}'),
    ('Known Drivers', f'{N_DRIVERS}'),
    ('dN/dS > 2 (drivers)', f'{n_sig_drivers}'),
    ('Hotspot Genes', f'{is_hotspot.sum()}'),
    ('Oncogenes', f'{n_onco}'),
    ('Tumor Suppressors', f'{n_tsg}'),
    ('Top Driver', f'{real_drivers[np.argmax(mutation_recurrence[driver_indices])]}'),
    ('Top Driver Freq', f'{mutation_recurrence[driver_indices].max()/N_TUMORS*100:.1f}%'),
    ('Pathways Enriched', f'{sum(1 for v in pathway_enrichment.values() if v > 1.5)}'),
]
y_pos = 0.95
for label, value in summary_lines:
    ax.text(0.05, y_pos, label + ':', color='#8b949e', fontsize=9, transform=ax.transAxes)
    ax.text(0.65, y_pos, value, color=ACCENT3, fontsize=9, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.09

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/cancer_driver_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()

import shutil
try:
    shutil.copy(__file__, '/mnt/shared-workspace/shared/cancer_driver_engine.py')
except shutil.SameFileError:
    pass  # already in destination

print("=== CancerDriverEngine Results ===")
print(f"Genes: {N_GENES}, Tumors: {N_TUMORS}, Drivers: {N_DRIVERS}")
print(f"Genes with dN/dS > 2: {n_sig_drivers}")
print(f"Hotspot genes: {is_hotspot.sum()}")
print(f"Oncogenes: {n_onco}, TSGs: {n_tsg}")
print(f"Top driver: {real_drivers[np.argmax(mutation_recurrence[driver_indices])]} ({mutation_recurrence[driver_indices].max()/N_TUMORS*100:.1f}%)")
print(f"Pathway enrichment: {dict(zip(pw_names, [round(v,2) for v in pw_scores]))}")
print(f"Dashboard saved: /mnt/shared-workspace/shared/cancer_driver_engine_dashboard.png")
