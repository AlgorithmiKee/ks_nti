import torch
import pandas

# q^2/r^2 in dB
range_nu_db = [0.0, -10.0, -20.0]
# 1/r^2 in dB
range_prec_r_db = [-10.0, 0.0, 10.0, 20.0, 30.0]

def export_ks_table(mse_path, std_path, tab_path):
    nu_idx =  [r'$\nu=0$ dB', r'$\nu=-10$ dB', r'$\nu=-20$ dB']
    cols = [r'$r^{-2}=-10$ dB', r'$r^{-2}=0$ dB', r'$r^{-2}=10$ dB', r'$r^{-2}=20$ dB', r'$r^{-2}=30$ dB']

    MSEs = torch.load(mse_path)
    stds = torch.load(std_path)

    df_mse = pandas.DataFrame(MSEs.numpy())
    df_mse.columns = cols
    df_mse['mo'] = ['MSE', 'MSE', 'MSE']
    df_mse['nu'] = nu_idx

    # df['m'] = pd.Categorical(df['m'], ["March", "April", "Dec"])
    
    df_std = pandas.DataFrame(stds.numpy())
    df_std.columns = cols    
    df_std['mo'] = ['std', 'std', 'std']
    df_std['nu'] = nu_idx


    df_mse_std = pandas.concat([df_mse, df_std]).sort_index(axis=0, ascending=False) # merge df
    df_mse_std = df_mse_std.groupby(['nu', 'mo']).mean() # multi index
    df_mse_std = df_mse_std.reindex(nu_idx, level=0).astype(float).round(3) # reindex & round
    df_mse_std = df_mse_std.rename_axis(index=(None, None), columns=None) # remove multiindex name

    table_ks = df_mse_std.to_latex(
        column_format = 'ccccccc',
        escape=False
    ).replace("\\\n$\\nu", "\\ \hline\n$\\nu")
    # cols = ['r^-2=-10 dB', 'r^-2=0 dB', 'r^-2=10 dB', 'r^-2=20 dB', 'r^-2=30 dB']
    # rows = ['nu=0 dB, MSE', 'nu=0 dB, std', 'nu=-10 dB, MSE', 'nu=-10 dB, std', 'nu=-20 dB, MSE', 'nu=-20 dB, std']
    # df_mse_std.index = rows
    # df_mse_std.columns = cols
    with open(tab_path, 'w') as tex_file:
        tex_file.write(table_ks)
    return df_mse_std

if __name__ == '__main__':
    mse_paths = [
        'Sim_baseline/KF/MSEs_state_baseline.pt',
        'Sim_baseline/KS/MSE_state_rts.pt',

        'Sim_const_R/MSE_state_unknownR.pt',
        'Sim_const_Q/MSE_state_unknownQ.pt',

        'Sim_sin_R/Pd100/baseline/MSE_state_rts.pt',
        'Sim_sin_R/Pd100/MA/MSE_state_unknownR.pt',
        'Sim_sin_R/Pd100/LF/MSE_state_unknownR.pt',

        'Sim_sin_Q/Pd100/baseline/MSE_state_rts.pt',
        'Sim_sin_Q/Pd100/MA/MSE_state_unknownQ.pt',
        'Sim_sin_Q/Pd100/LF/MSE_state_unknownQ.pt'        
    ]
    std_paths = [
        'Sim_baseline/KF/std_MSE_state_baseline.pt',
        'Sim_baseline/KS/std_MSE_state_rts.pt',

        'Sim_const_R/std_MSE_state_unknownR.pt',
        'Sim_const_Q/std_MSE_state_unknownQ.pt',

        'Sim_sin_R/Pd100/baseline/std_MSE_state_rts.pt',
        'Sim_sin_R/Pd100/MA/std_MSE_state_unknownR.pt',
        'Sim_sin_R/Pd100/LF/std_MSE_state_unknownR.pt',

        'Sim_sin_Q/Pd100/baseline/std_MSE_state_rts.pt',
        'Sim_sin_Q/Pd100/MA/std_MSE_state_unknownQ.pt',
        'Sim_sin_Q/Pd100/LF/std_MSE_state_unknownQ.pt'     
    ]
    tab_paths = [
        'Sim_baseline/KF/table.tex',
        'Sim_baseline/KS/table.tex',

        'Sim_const_R/table.tex',
        'Sim_const_Q/table.tex',

        'Sim_sin_R/Pd100/baseline/table.tex',
        'Sim_sin_R/Pd100/MA/table.tex',
        'Sim_sin_R/Pd100/LF/table.tex',

        'Sim_sin_Q/Pd100/baseline/table.tex',
        'Sim_sin_Q/Pd100/MA/table.tex',
        'Sim_sin_Q/Pd100/LF/table.tex'    
    ]

    for mse, std, tab in zip(mse_paths, std_paths, tab_paths):
        export_ks_table(mse, std, tab)