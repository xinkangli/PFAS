"""
common plotting style for stage 4-6 figures.
Times New Roman (or DejaVu Serif fallback), Nature/子刊-ish palette,
slightly larger fonts, generous margins to keep text from overlapping plots.
"""
from matplotlib import pyplot as plt
import matplotlib as mpl

# Try Times New Roman; fall back gracefully
import matplotlib.font_manager as fm
TIMES = None
for cand in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Nimbus Roman', 'serif'):
    try:
        fm.findfont(cand, fallback_to_default=False)
        TIMES = cand
        break
    except Exception:
        continue
TIMES = TIMES or 'serif'

mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : [TIMES],
    'mathtext.fontset': 'stix',
    'font.size'       : 13,
    'axes.titlesize'  : 15,
    'axes.labelsize'  : 14,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
    'legend.fontsize' : 11,
    'figure.titlesize': 16,
    'figure.dpi'      : 130,
    'savefig.dpi'     : 300,
    'savefig.bbox'    : 'tight',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.linewidth'  : 1.0,
    'pdf.fonttype'    : 42,
    'ps.fonttype'     : 42,
})

# Nature Communications-ish qualitative palette (color-blind friendly)
NC_PALETTE = [
    '#0C5DA5',  # blue
    '#FF2C00',  # red
    '#00B945',  # green
    '#FF9500',  # orange
    '#845B97',  # purple
    '#474747',  # dark grey
    '#9E9E9E',  # grey
    '#56B4E9',  # light blue
    '#E69F00',  # gold
    '#F0E442',  # yellow
    '#CC79A7',  # pink
    '#009E73',  # teal
]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=NC_PALETTE)

# continuous palettes
SEQ_CMAP   = 'viridis'          # for predictions
DIV_CMAP   = 'RdBu_r'           # for risk diverging
PERSIST_C  = {'vP': '#FF2C00', 'P': '#FF9500', 'not-P_self': '#00B945',
              'not-P': '#00B945', 'NP': '#9E9E9E'}

def neat_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4, width=1.0)

print(f'[style] using font {TIMES!r}')
