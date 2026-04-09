from matplotlib import rc

rc('text',       usetex=True)                       # Use latex
rc('text.latex', preamble=r"\usepackage{fourier}")
rc('font',       family='serif', size=12)
rc('grid',       linestyle='dotted')
# rc('xtick',      top=False, bottom=False)
# rc('xtick.major',pad=0)
# rc('ytick',      left=False, right=False)
# rc('ytick.major',pad=0)
rc('legend',     fontsize='small', handlelength=1.5, handletextpad=0.5, borderpad=0.2, frameon=True, columnspacing=0.75)
# rc('figure',     titlesize='medium')
# rc('axes',       grid=True)
rc('axes',       titlesize='medium')

figsize11=(4,3)
figsize12=(10,3)
figsize22=(10,6)
figsize21=(6,3)
figsize41=(12,3)

# At home, screen 27''
rc('figure',     dpi=200)
rc('savefig',    dpi=100)
