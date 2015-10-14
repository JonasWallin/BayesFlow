import numpy as np
from matplotlib.pyplot import get_cmap
import colorsys


def black_ip(color, n, N):
    '''
        Interpolate between a color and black.

        color   -   color
        n       -   shade rank
        N       -   total number of possible shades
    '''
    if len(color) == 4:
        r, g, b, al = color
    else:
        r, g, b = color
        al = 1
    h, s, v = colorsys.rgb_to_hsv(r,  g,  b)
    r, g, b = colorsys.hsv_to_rgb(h, s, (N+3-n)*1./(N+3)*v)
    return r, g, b, al


def major_colors(nbr_colors):
    cm = get_cmap('gist_rainbow')
    return [cm(((s+.4) % nbr_colors)*1./(nbr_colors-1)) for s in range(nbr_colors)]


def get_colors(sucos, suco_ord, maxcol=8):
    nbr_colors = min(maxcol, len(suco_ord))
    maj_colors = major_colors(nbr_colors)
    suco_col = [maj_colors[k] if k < nbr_colors else
                maj_colors[k % nbr_colors][:3]+(.5,) for k in np.argsort(suco_ord)]
        # transparent colors after nbr_colors
    nbr_comp = sum(len(suco) for suco in sucos)
    comp_col = [(0, 0, 0)]*nbr_comp

    for col, suco in zip(suco_col, sucos):
        for i, k in enumerate(suco):
            comp_col[k] = black_ip(col, i, len(suco))
    return comp_col, suco_col
