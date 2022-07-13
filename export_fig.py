import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from param_lin import range_prec_r_db

NORMAL_TICK = 10
SMALL_TICK = 6

def save_ksplot_no_baseline(ipath, opath, title, ilabel):
    # load data 
    MSEs_non_base = torch.load(ipath)

    ##########################################
    #### Plot MSE of input for unknown R ####
    ##########################################
    plt.figure()
    if title is not None:
        plt.title(title)

    plt.xlabel(r'$\frac{1}{r^2}$ [dB]')
    plt.ylabel('MSE [dB]')
    
    # Noise floor
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    # plot MSE 
    plt.plot(range_prec_r_db, MSEs_non_base[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$=0 dB, ' + ilabel)
    plt.plot(range_prec_r_db, MSEs_non_base[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$=-10 dB, ' + ilabel)
    plt.plot(range_prec_r_db, MSEs_non_base[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$=-20 dB, ' + ilabel)
    
    plt.legend(loc="upper right")
    plt.grid()

    plot_name = ''
    if 'input' in ipath:
        plot_name = 'MSE_input'
    elif 'state' in ipath:
        plot_name = 'MSE_state'
    
    plt.savefig(opath + plot_name + '.pdf')
    plt.close()

# ipath: string var, path of MSEs (non base simulation)
# bpath: string var, path of MSEs (baseline simulation)
# opath: string var, path of output file
# title: string var, tile of the plot
# blabel: string var, baseline legend
# ilabel: string var, non base legend
# zoom_in: bool var, plot zoomed in x-axis
def save_ksplot(ipath, opath, bpath, ilabel, blabel=r'KS'):
    # plot name and zoom_in factor
    plot_name = ''
    zoomplt_scale_x, zoomplt_scale_y = .9, .9
    if 'input' in ipath:
        plot_name = 'MSE_input'
        zoomplt_scale_x, zoomplt_scale_y = .8, .8
    elif 'state' in ipath:
        plot_name = 'MSE_state'
    
    # load data
    MSEs_baseline = torch.load(bpath)    
    MSEs_non_base = torch.load(ipath)

    # main ax and zoomed in ax
    fig, axmain = plt.subplots(figsize=(8,5))
    axins = inset_axes(axmain, width="40%", height="40%", bbox_to_anchor=(.05, .05, .9, .9), bbox_transform=axmain.transAxes, loc=3)

    # plot data
    for ax in [axmain, axins]:
        # plot noise floor
        ax.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
        # plot for nu = 0 dB
        ax.plot(range_prec_r_db, MSEs_non_base[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$=0 dB, ' + ilabel)
        ax.plot(range_prec_r_db, MSEs_baseline[0, :], '--', c='c', linewidth=0.75, label=r'$\nu$=0 dB, ' + blabel)
        # plot for nu = -10 dB
        ax.plot(range_prec_r_db, MSEs_non_base[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$=-10 dB, ' + ilabel)
        ax.plot(range_prec_r_db, MSEs_baseline[1, :], '--', c='b', linewidth=0.75, label=r'$\nu$=-10 dB, ' + blabel)
        # plot for nu = -20 dB
        ax.plot(range_prec_r_db, MSEs_non_base[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$=-20 dB, ' + ilabel)
        ax.plot(range_prec_r_db, MSEs_baseline[2, :], '--', c='g', linewidth=0.75, label=r'$\nu$=-20 dB, ' + blabel)

        
    # layout main axes
    axmain.set_xlabel(r'$\frac{1}{r^2}$ [dB]')
    axmain.set_ylabel('MSE [dB]')
    axmain.tick_params(axis='x', labelsize=NORMAL_TICK)
    axmain.tick_params(axis='y', labelsize=NORMAL_TICK)    
    axmain.grid(linewidth=0.5)
    axmain.legend(loc='upper right')
    
    # layout zoomed axes
    axins.tick_params(axis='x', labelsize=SMALL_TICK)
    axins.tick_params(axis='y', labelsize=SMALL_TICK)
    axins.grid(linewidth=0.25)
    x_min, x_max = -0.5, 0.5
    y_min, y_max = min(MSEs_baseline[2, 1], MSEs_non_base[2, 1])-1.0, -range_prec_r_db[1]+1.0 
    axins.set_xlim(x_min, x_max) 
    axins.set_ylim(y_min, y_max) 

    mark_inset(axmain, axins, loc1=1, loc2=2, fc="none", ec="0.5", lw=0.5)
    
    plt.savefig(opath + plot_name + '.pdf')
    plt.close()


if __name__ == '__main__':
    #########################
    ####   KS versus KF  ####
    #########################    
    bpath = 'Sim_baseline/KF/MSEs_state_baseline.pt'
    ipath = 'Sim_baseline/KS/MSE_state_rts.pt'
    opath = 'Sim_baseline/KS/'
    save_ksplot(ipath, opath, bpath, ilabel='KS', blabel='KF')
    # save_ksplot_no_baseline(ipath, opath,   ilabel='KS')

    #########################
    #### unknown const Q ####
    #########################
    bpath = 'Sim_baseline/KS/MSE_state_rts.pt'
    ipath = 'Sim_const_Q/MSE_state_unknownQ.pt'
    opath = 'Sim_const_Q/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS, unknown $Q$', blabel=r'KS, known $Q$')

    bpath = 'Sim_baseline/KS/MSE_input_rts.pt'
    ipath = 'Sim_const_Q/MSE_input_unknownQ.pt'
    opath = 'Sim_const_Q/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS, unknown $Q$', blabel=r'KS, known $Q$')

    #########################
    #### unknown const R ####
    #########################
    bpath = 'Sim_baseline/KS/MSE_state_rts.pt'
    ipath = 'Sim_const_R/MSE_state_unknownR.pt'
    opath = 'Sim_const_R/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS, unknown $R$', blabel=r'KS, known $R$')

    bpath = 'Sim_baseline/KS/MSE_input_rts.pt'
    ipath = 'Sim_const_R/MSE_input_unknownR.pt'
    opath = 'Sim_const_R/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS, unknown $R$', blabel=r'KS, known $R$')

    ############################
    #### unknown sin Q (MA) ####
    ############################
    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_state_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/MA/MSE_state_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/MA/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+MA, unknown $Q$', blabel=r'KS, known $Q$')

    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_input_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/MA/MSE_input_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/MA/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+MA, unknown $Q$', blabel=r'KS, known $Q$')

    ############################
    #### unknown sin R (MA) ####
    ############################
    bpath = 'Sim_sin_R/Pd100/baseline/MSE_state_rts.pt'
    ipath = 'Sim_sin_R/Pd100/MA/MSE_state_unknownR.pt'
    opath = 'Sim_sin_R/Pd100/MA/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+MA, unknown $R$', blabel=r'KS, known $R$')

    bpath = 'Sim_sin_R/Pd100/baseline/MSE_input_rts.pt'
    ipath = 'Sim_sin_R/Pd100/MA/MSE_input_unknownR.pt'
    opath = 'Sim_sin_R/Pd100/MA/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+MA, unknown $R$', blabel=r'KS, known $R$')


    ############################
    #### unknown sin Q (LF) ####
    ############################
    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_state_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/LF/MSE_state_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/LF/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $Q$', blabel=r'KS, known $Q$')

    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_input_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/LF/MSE_input_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/LF/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $Q$', blabel=r'KS, known $Q$')

    ############################
    #### unknown sin R (LF) ####
    ############################
    bpath = 'Sim_sin_R/Pd100/baseline/MSE_state_rts.pt'
    ipath = 'Sim_sin_R/Pd100/LF/MSE_state_unknownR.pt'
    opath = 'Sim_sin_R/Pd100/LF/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $R$', blabel=r'KS, known $R$')

    bpath = 'Sim_sin_R/Pd100/baseline/MSE_input_rts.pt'
    ipath = 'Sim_sin_R/Pd100/LF/MSE_input_unknownR.pt'
    opath = 'Sim_sin_R/Pd100/LF/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $R$', blabel=r'KS, known $R$')
