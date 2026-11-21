"""Likelihood geometry with a toy SIE lens, a GRF draw, GP and measurement noise.
Uses the draft's beta_i = backprojection(theta_i_true) convention.
PDF only. Error scales are exaggerated; this is not a fitted cluster.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from make_likelihood_diagram import build, back_project, ellipse, arrow, SIGMA_MEAS, INK, MUTED, MEAS, GP, GRF, CRITICAL_COLOR

OUT = Path(__file__).resolve().parents[1]/'paper/figures/likelihood_caustic_diagram.pdf'


def mark(ax, p, text, offset, marker='o', color=INK, filled=False):
    ax.plot(*p, marker=marker, ms=8 if marker!='*' else 13, mec=color,
            mfc=color if filled else 'white', mew=1.4, zorder=8)
    ax.annotate(text, p, xytext=offset, textcoords='offset points', fontsize=11,
                color=color, ha='left', va='center', zorder=9,
                arrowprops={'arrowstyle':'-', 'lw':.55, 'color':MUTED, 'shrinkB':6})


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'axes.grid':False,'pdf.fonttype':42,'font.size':10})
    t = build()
    i=t['highlight']
    b=back_project(t['true']); bm=back_project(t['measured']); b0=back_project(t['lens'])
    fig=plt.figure(figsize=(12,12))
    ia=fig.add_axes([.07,.57,.35,.36]); sa=fig.add_axes([.59,.57,.35,.36])
    iz=fig.add_axes([.07,.20,.35,.29]); sz=fig.add_axes([.59,.20,.35,.29])
    for ax in (ia,sa,iz,sz):
        ax.set_aspect('equal')
        ax.spines[['top','right']].set_visible(False)
    ia.plot(*t['critical'].T,color=CRITICAL_COLOR,lw=1.4)
    ia.plot(*t['measured'].T,'x',color=INK,ms=6)
    for j,p in enumerate(t['measured']):
        ia.annotate(str(j+1),p,xytext=(7,7),textcoords='offset points')
    ia.set(xlim=(-30,30),ylim=(-30,30),xlabel=r'Image plane: $\theta_1$ (arcsec)',ylabel=r'$\theta_2$ (arcsec)')
    ia.text(.03,.95,'Critical curve',transform=ia.transAxes,color=CRITICAL_COLOR,va='top')
    sa.plot(*t['caustic'].T,color=CRITICAL_COLOR,lw=1.4)
    sa.plot(*b.T,'o',mfc='white',mec=INK,ms=5)
    sa.plot(*t['source'],marker='*',color=INK,ms=11)
    cusp=np.abs(t['caustic']).max()*1.15
    sa.set(xlim=(-cusp,cusp),ylim=(-cusp,cusp),xlabel=r'Source plane: $\beta_1$ (arcsec)',ylabel=r'$\beta_2$ (arcsec)')
    sa.text(.03,.95,'Diamond caustic',transform=sa.transAxes,color=CRITICAL_COLOR,va='top')
    sa.annotate(r'$\beta_k$',t['source'],xytext=(-23,15),textcoords='offset points')
    fig.text(.50,.79,r'$\beta(\theta)=\theta-\alpha(\theta)$',ha='center',fontsize=12)
    fig.text(.50,.75,'Lens mapping',ha='center',color=MUTED)
    arrow(fig.add_axes([.445,.70,.11,.04],frameon=False),(.05,.5),(.95,.5),MUTED)
    fig.axes[-1].set(xticks=[],yticks=[],xlim=(0,1),ylim=(0,1))

    # Image zoom: GRF has already perturbed the lens-equation root theta_0.
    p0,pt,pm=t['lens'][i],t['true'][i],t['measured'][i]
    ctr=(p0+pt+pm)/3
    iz.set(xlim=(ctr[0]-.40,ctr[0]+.40),ylim=(ctr[1]-.32,ctr[1]+.32),
           xlabel=r'$\theta_1$ (arcsec)',ylabel=r'$\theta_2$ (arcsec)')
    e=ellipse(SIGMA_MEAS**2*np.eye(2),color=MEAS,lw=1.6);e.set_center(pt);iz.add_patch(e)
    e=ellipse(.08**2*np.eye(2),color=GP,lw=1.4,ls='--');e.set_center(p0);iz.add_patch(e)
    arrow(iz,p0,pt,GP);arrow(iz,pt,pm,MEAS)
    mark(iz,p0,r'$\theta_i^0$',(13,19),marker='+')
    mark(iz,pt,r'$\theta_i^{\rm true}$',(-48,-24))
    mark(iz,pm,r'$\theta_i^{\rm meas}$',(-66,22),marker='x')
    iz.text(.02,.98,f'Image {i+1}: local view',transform=iz.transAxes,va='top',fontsize=10)

    # Source zoom: the latent backprojection is beta_i, NOT beta(theta_meas).
    pk,pr,pi,pme=t['source'],b0[i],b[i],bm[i]
    ctr=(pk+pr+pi+pme)/4
    half=.29
    sz.set(xlim=(ctr[0]-half,ctr[0]+half),ylim=(ctr[1]-half*.8,ctr[1]+half*.8),
           xlabel=r'$\beta_1$ (arcsec)',ylabel=r'$\beta_2$ (arcsec)')
    for key,c,center,ls in [('c_grf',GRF,pk,'-'),('c_gp',GP,pr,'--'),('c_meas',MEAS,pi,':')]:
        e=ellipse(t[key][i,:,i,:],color=c,lw=1.5,ls=ls);e.set_center(center);sz.add_patch(e)
    arrow(sz,pk,pr,GRF);arrow(sz,pr,pi,GP);arrow(sz,pi,pme,MEAS)
    mark(sz,pk,r'$\beta_k$',(12,15),marker='*',filled=True)
    mark(sz,pr,r'$\beta(\theta_i^0)$',(-93,-34),marker='+')
    mark(sz,pi,r'$\beta_i=\beta(\theta_i^{\rm true})$',(45,8))
    mark(sz,pme,r'$\beta(\theta_i^{\rm meas})$',(-96,4),marker='x')
    sz.text(.02,.98,r'$F_i=\beta_i-\beta_k$',transform=sz.transAxes,va='top',fontsize=12)
    # Locate the enlarged regions in the two global planes.
    ia.indicate_inset_zoom(iz,edgecolor=MUTED,alpha=.45)
    sa.indicate_inset_zoom(sz,edgecolor=MUTED,alpha=.45)
    fig.text(.50,.43,'Same image,\nbefore and after\nbackprojection',ha='center',color=MUTED,fontsize=10)
    handles=[Line2D([],[],color=c,lw=1.8,ls=ls) for c,ls in [(MEAS,':'),(GP,'--'),(GRF,'-')]]
    fig.legend(handles,[r'Measurement: $C_{{\rm meas},ii}=\sigma_{{\rm meas},i}^2 A_iA_i^{\rm T}$',
                        r'GP displacement: $C_{{\rm GP},ij}=A_iK_{{\rm GP},ij}A_j^{\rm T}$',
                        r'GRF deflection: $C_{{\rm GRF},ij}=K_\alpha(\theta_i-\theta_j)$'],
               loc='upper center',bbox_to_anchor=(.50,.17),ncol=1,frameon=False,fontsize=10)
    fig.text(.5,.055,r'$\mathcal{L}_i=\int d^2\theta_i^{\rm true}\;'
             r'\mathcal{N}(\theta_i^{\rm meas}-\theta_i^{\rm true};\sigma_{{\rm meas},i}^2 I)\;'
             r'\mathcal{N}(\beta_i-\beta_k;C_{{\rm model},ii})\;|\det A_i|$',ha='center',fontsize=12)
    fig.text(.5,.023,r'$C_{\rm model}=C_{\rm GRF}+C_{\rm GP}$. Contours: unit Mahalanobis radius. Toy lens; exaggerated errors.',
             ha='center',color=MUTED,fontsize=9)
    # Reserve a separate band for the complete coupled likelihood.
    for ax in fig.axes:
        bounds=ax.get_position()
        ax.set_position([bounds.x0,.12+.88*bounds.y0,bounds.width,.88*bounds.height])
    for text in fig.texts:
        x,y=text.get_position();text.set_position((x,.12+.88*y))
    fig.legends[0].set_bbox_to_anchor((.50,.12+.88*.17))
    fig.text(.5,.109,r'$\theta_i^0$: image position in the halo + GRF realization, before GP displacement.',ha='center',fontsize=9,color=MUTED)
    fig.text(.5,.085,
             r'$\mathcal{L}_{\rm pos}(\Theta)=\int d^{2N_{\rm im}}\theta^{\rm true}\;'
             r'\left[\prod_i\mathcal{N}(\theta_i^{\rm meas}-\theta_i^{\rm true};\sigma_{{\rm meas},i}^{2}I)\right]'
             r'\left[\mathcal{N}(F(\theta^{\rm true});C_{\rm model})\prod_i|\det A(\theta_i^{\rm true})|\right]$',
             ha='center',fontsize=12)
    fig.text(.34,.053,'Measurement factor: independent image errors',ha='center',color=MEAS,fontsize=10)
    fig.text(.76,.053,'Model factor: correlated residuals + area conversion',ha='center',color=GRF,fontsize=10)
    fig.text(.5,.022,'Integrate over every latent true image position; sample the family source positions together with the lens parameters.',ha='center',fontsize=9,color=MUTED)
    OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT);plt.close(fig)
    print(OUT)
    print('image',i+1,'positions',pk,pr,pi,pme)

if __name__=='__main__':main()
