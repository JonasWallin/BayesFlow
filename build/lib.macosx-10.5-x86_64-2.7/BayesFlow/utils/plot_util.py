from __future__ import division
from matplotlib.pyplot import get_cmap
import colorsys

def black_ip(color,n,N):
    '''
        Interpolate between a color and black.
        
        color   -   color
        n       -   shade rank
        N       -   total number of possible shades
    '''
    if len(color) == 4:
        r,g,b,al = color
    else:
        r,g,b = color
        al = 1
    h,s,v = colorsys.rgb_to_hsv(r, g, b)
    r,g,b = colorsys.hsv_to_rgb(h,s,(N+3-n)/(N+3)*v)
    return r,g,b,al
    #return colorsys.hsv_to_rgb(h,s,(N+3-n)/(N+3)*v) 

def get_colors(sucos,suco_ord,comp_ord,maxcol=8):
        nbrsucocol = min(maxcol,len(suco_ord))  
        suco_col = [(0,0,0)]*len(suco_ord)
        comp_col = [(0,0,0)]*len(comp_ord)
        sucos_sort = [sucos[i] for i in suco_ord]
        cm = get_cmap('gist_rainbow')
        for s,suco in enumerate(sucos_sort):
            #print "(s % nbrsucocol)/nbrsucocol = {}".format((s % nbrsucocol)/nbrsucocol)
            suco_col[suco_ord[s]] = cm((s % nbrsucocol)/nbrsucocol)
            if s > maxcol:
                suco_col[suco_ord[s]] = suco_col[suco_ord[s]][:3]+(0.5,)
            for i,k in enumerate(suco):
                comp_col[k] = black_ip(suco_col[suco_ord[s]],i,len(suco))
        return comp_col,suco_col  