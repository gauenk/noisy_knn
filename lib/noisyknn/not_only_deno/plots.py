
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- io --
from pathlib import Path

# -- plotting --
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def compare_psnr_vs_pme(records,sname):


    #
    # --- Plotting Info ---
    #

    # -- CONSTS --
    FSIZE = 14
    FSIZE_B = 16
    FSIZE_S = 12

    # -- formatting --
    colors = np.array(['r','g','b','y'])

    #
    # --- Aggregate ---
    #

    # -- create figure --
    ginfo = {'width_ratios': [1,],'wspace':0, 'hspace':0.1,
             "top":0.9,"bottom":0.20,"left":0.20,"right":0.96}
    fig,ax = plt.subplots(figsize=(4,3),gridspec_kw=ginfo)
    ax = [ax]

    # -- aggregate  --
    print(records.columns)
    print(records['psnrs'])
    records_agg = records.groupby(["seed"]).mean()
    print(records_agg.columns)
    records_agg = records.groupby(["k","seed"]).mean()
    print(records_agg.columns)
    psnrs_agg = records_agg['psnrs'].to_numpy()
    pmin,pmax = psnrs_agg.min().item(),psnrs_agg.max().item()


    # -- plot each "k" [num of neighbors] sep --
    kidx = 0
    for k,kdf in records_agg.groupby("k"):

        # -- unpack --
        pme = kdf['pme'].to_numpy()
        psnrs = kdf['psnrs'].to_numpy()
        pme_clean = kdf['pme_clean'].to_numpy()
        pme_noisy = kdf['pme_noisy'].to_numpy()

        # -- unpack fmt --
        color = colors[kidx]

        # -- plot data --
        cs = ax[0].scatter(pme,psnrs,c=color,label=k,marker="o")

        # -- plot clean --
        pmean = psnrs.mean()
        c_u_name = "c_%s"%k
        c_pme = pme_clean.mean()
        # cs = ax[0].vlines(c_pme,ymin=pmin,ymax=pmax,color="r")
        # cs = ax[0].scatter(c_pme,psnrs.mean(),color=color,marker='+')
        # cs = ax[0].scatter(c_pme,pmin,color=color,marker='+')

        # -- plot noisy --
        n_u_name = "n_%s"%k
        n_pme = pme_noisy.mean()
        # cs = ax[0].vlines(n_pme,ymin=pmin,ymax=pmax,color="r")
        # cs = ax[0].scatter(n_pme,psnrs.mean(),color=color,marker='x')
        # cs = ax[0].scatter(n_pme,pmin,color=color,marker='x')

        # -- plot clean -> noisy bar --
        cs = ax[0].plot([c_pme,n_pme],[pmean,pmean],c=color,marker='|',
                        markersize=50)

        # -- update iterate --
        kidx += 1

    # -- update x ticks --
    pme_agg = records_agg['pme'].to_numpy()
    pme_ticks = np.linspace(pme_agg.min(),pme_agg.max(),3)
    # pme_ticks = np.quantile(pme_agg,[0.15,0.5,0.85])
    pme_ticklabels = ["%1.3f" %x for x in pme_ticks]
    ax[0].set_xticks(pme_ticks)
    ax[0].set_xticklabels(pme_ticklabels,fontsize=FSIZE_S)

    # -- update y ticks --
    psnrs_agg = records_agg['psnrs'].to_numpy()
    psnrs_ticks = np.linspace(psnrs_agg.min(),psnrs_agg.max(),3)
    # psnrs_ticks = np.quantile(psnrs_agg,[0.15,0.5,0.85])
    psnrs_ticklabels = ["%2.1f" %x for x in psnrs_ticks]
    ax[0].set_yticks(psnrs_ticks)
    ax[0].set_yticklabels(psnrs_ticklabels,fontsize=FSIZE_S)

    # handles,labels = ax[0].get_legend_handles_labels()
    # cs = ax[0].scatter(pme_clean,psnrs,c='k')
    # cs = ax[1].scatter(pme_clean,psnrs,c=colors[names])
    # cs = ax[1].plot(pme_clean,pme,'x')
    # if sname == "real": title,x = "Real-World Motion [Set8]",0.44
    # elif sname == "sim": title,x = "Simulated Global Motion [1ppf]",0.44
    # else: raise ValueError(f"Uknown sname [{sname}]")
    title,x = "Hi",0.44
    ax[0].set_title(title,fontsize=FSIZE_B,x=x)
    ax[0].set_xlabel("Patch Matching Error",fontsize=FSIZE_B)
    ax[0].set_ylabel("PSNR",fontsize=FSIZE_B)

    # -- add legend [num neighs] --
    # leg1 = ax[0].legend(bbox_to_anchor=(.98,1), loc="upper left",fontsize=FSIZE_S,
    #                     title="Neighs.",title_fontsize=FSIZE_S,framealpha=0.)
    leg1 = ax[0].legend(bbox_to_anchor=(.65,.75), loc="upper left",fontsize=FSIZE,
                        title="Neighs.",title_fontsize=FSIZE,framealpha=0.)
    ax[0].add_artist(leg1)

    # -- add legend [] --
    # h_0 = plt.plot([],[], color="k", marker="+", ls="")[0]
    # h_1 = plt.plot([],[], color="k", marker="o", ls="")[0]
    # h_2 = plt.plot([],[], color="k", marker="x", ls="")[0]
    # hands = [h_0,h_1,h_2]
    # hands_lab = ['Clean','Modified','Noisy']
    # leg2 = plt.legend(hands, hands_lab, loc="upper left",
    #                   bbox_to_anchor=(0.22,.80),
    #                   title="Image",title_fontsize=FSIZE,
    #                   fontsize=FSIZE, framealpha=0.)
    # ax[0].add_artist(leg2)

    # -- save --
    path = Path("/home/gauenk/Documents/packages/noisy_knn/output/not_only_deno")
    if not path.exists(): path.mkdir()
    save_name = "scatter_%s" % sname
    fn = "not_only_deno_%s.png" % save_name
    fn = path / fn
    plt.savefig(fn,dpi=300,transparent=True)

    # -- close --
    plt.close("all")
    plt.cla()
    plt.clf()
