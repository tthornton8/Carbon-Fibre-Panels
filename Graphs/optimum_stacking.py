import numpy as np
import pandas as pd

def cfrp_stiffness(tk, phi = np.array([0, 45, 90]), E = 130):
    """ Use Krenchel's method to calculate panel effective modulus. """

    cos_4_phi = np.cos(phi * (np.pi/180))**4
    tk_pc = tk/tk.sum()
    return np.sum(tk_pc * cos_4_phi)*E

def dec2bin(x, n = 3):
    """ Return the binary representation of x as a list, padding to a number of digits, n. """

    x = bin(x)
    b = x + '0'*((n+2)-len(x))
    b = b[2:]
    b = [1-2*int(i) for i in b]
    return b
    

def stack(seq, t_panel, t = 0.125, tolerance = 0.08):
    """ Generate a lost of all possible stacking sequences for a given target ply distribution, panel thickness, ply thickness, and tolerance against the atrget distribution """

    # create blank dataframe with columns for the sequence, its modulus, and thickness.
    df = pd.DataFrame(columns=['seq', 'E', 't'])

    # generate the 'perfect' stacking sequence (this will result in a non-integer number of plies in most cases).
    seq = np.array(seq)/100
    perfect = seq*t_panel/t

    # round the 'perfect' sequence to a whole number of plies in each direction
    rounded = np.round(perfect)

    # find where the number of plies is odd, as they must be even
    combinations = rounded%2

    # combination of plies that are allowed to be od (one of 0 degrees or 90 degrees)
    allowable_odds = [np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([-1,0, 0])]

    # what to add to each rounded ply value to make them a whole number, accounting for all posible combinations of addition and subtraction
    combinations = [dec2bin(i)*combinations for i in range(8)]

    # allow some plies to be odd
    combinations = combinations + [i * np.array([1, 1, 0]) for i in combinations] +  [i * np.array([0, 1, 1]) for i in combinations]
    for j in allowable_odds:
        combinations = combinations + [j+i for i in combinations]

    # add the combinations to the rounded ply values to get all possible combinations of stacking sequences
    combinations = [rounded + i for i in combinations]

    # ensure combinations obey stacking sequence rules
    combinations = [i for i in combinations if (not i[1]%2)]
    combinations = [i + np.array([0, 2, 0]) if (i[1]%4) else i for i in combinations] + [i + np.array([0, -2, 0]) if (i[1]%4) else i for i in combinations]
    combinations = [i for i in combinations if not any(i<=0)]
    combinations = [i for i in combinations if i[2] > 2]
    combinations = [i for i in combinations if (not i[0] % 2) or (not i[2] % 2)]
    combinations = [i for i in combinations if i[2]/sum(i) < 17]

    # get all unique remaining combinations
    combinations = np.unique(combinations, axis = 0)
    
    # calculate the stiffness and thickness of each cvombination
    e_combinations = [cfrp_stiffness(i) for i in combinations]
    n_combinations = [sum(i) for i in combinations]

    # append each combination to the dataframe created earlier
    for i, (x, e, n) in enumerate(zip(combinations, e_combinations, n_combinations)):
        # ensure that stacking sequence will not result in more than 3 of the same ply adjacent
        check = x[0]/(x[1]-4 + x[2])

        # string representation of stacking sequence
        x_str = '/'.join(str(int(i)) for i in x)

        # ensure tolerance is met
        x_pc = x/x.sum()
        x_pc = [np.round(abs(i-j), 2) for i,j in zip(x_pc[:2], seq[:2])]
        if max(x_pc) < tolerance and check <= 6:

            # if all above conditions are met, append combination to dataframe
            df = df.append(pd.DataFrame([[x_str, e, n*t]], columns = ['seq', 'E', 't'], index = [i]))

    return df

def optimum_stacking(best_tsk, best_tst, best_bst, bsk = 180, kst = 1.65, ksk = 2.23, Nx = 1.74):
    """ return the optimum combination of stacking sequences for the skin and stiffener given target bst, tst, tsk """

    ### DEFINING FUNCTIONS INSIDE THE FUNCTION TO USE THE LOCAL SCOPE OF THE ENCASING FUNCTION SO VALUES DO NOT NEED TO BE PASSED AROUND ###
    def etbar_calc_df(row):
        tst = row['tst']
        tsk = row['tsk']
        Esk = row['Esk']
        Est = row['Est']
        etbar = Esk*tsk + Est*tst*(70+3*best_bst)/bsk
        return etbar

    def Nx_st_calc_df(row):
        etbar = row['Etbar']
        tst = row['tst']
        return etbar*kst*(tst/best_bst)**2

    def Nx_sk_calc_df(row):
        etbar = row['Etbar']
        tsk = row['tsk']

        bskc = best_bst*(1+2*np.cos(np.pi/3)) +35
        bsko = bsk-bskc

        return etbar*ksk*(tsk/max([bskc, bsko]))**2

    def Nx_euler_calc_df(row):
        tst = row['tst']
        tsk = row['tsk']
        Esk = row['Esk']
        Est = row['Est']
        bst = best_bst

        f1 = tsk/tst #LHS/RHS

        f4 = (Esk*f1*bsk + Est*(3*bst+70))/Est
        f2 = (1+f1) *(35+1.5*bst)
        f3 = np.sqrt(3)*bst**2

        a1 = f2/f4
        a2 = f3/f4

        def Nx(tst):
            zbar = a1*tst+a2

            b1 = zbar - tst*(1+f1)/2

            EI_skin = Esk*(bsk*tsk*zbar**2 + bsk*(f1*tst)**(3)/12)
            EI_3mm  = 70*(Esk*tst*(b1)**2 + Est*(tst**3)/12)
            EI_r    = Est*(tst*bst*(bst*np.sqrt(3)/2 - b1)**2 + bst*(tst**3)/12)
            EI_w    = Est*tst*((bst**3)/8 + 2*bst*(bst*np.sqrt(3)/4 - b1)**2)

            EI_tot = EI_skin + EI_3mm + EI_r + EI_w
            N_xeuler = np.pi**2/(800**2*bsk) * EI_tot
            
            return N_xeuler

        return Nx(tst)

    def mass_calc_df(row):
        tst = row['tst']
        tsk = row['tsk']
        bst = best_bst

        rho = 1600/(1e3**3) #kg/mm^3

        skin    = bsk*tsk   #mm^2
        st_3mm  = 2*35*tst  #mm^2
        st      = 3*bst*tst #mm^2

        mass = (skin+st_3mm+st)*rho #kg/mm
        mass = mass/bsk #kg/mm2
        mass = 1e6*mass #kg/m2
        return mass  

    # stacking sequences for skin and stiffener
    tsk_options = stack([44,44,12], best_tsk)
    tst_options = stack([60,30,10], best_tst, tolerance = 0.15)
    
    # all combinations of stacking sequences
    tst_options['key'] = '0'
    tsk_options['key'] = '0'
    options = tst_options.merge(tsk_options, on = 'key', suffixes=('st', 'sk'))
    options = options[[i for i in options.columns if i != 'key']]

    # calculate some reserve factors for each combination
    options['Etbar'] = options.apply(etbar_calc_df, axis = 1)
    options['Nx_st'] = options.apply(Nx_st_calc_df, axis = 1)
    options['Nx_sk'] = options.apply(Nx_sk_calc_df, axis = 1)
    options['Nx_mat'] = options.apply(lambda row: row['Etbar']*0.0045, axis = 1)
    options['Nx_euler'] = options.apply(Nx_euler_calc_df, axis = 1)
    options['mass'] = options.apply(mass_calc_df, axis = 1)
    options['RF_st'] = options['Nx_st']/Nx
    options['RF_sk'] = options['Nx_sk']/Nx
    options['RF_mat'] = options['Nx_mat']/Nx
    options['RF_euler'] = options['Nx_euler']/Nx

    # are all the reserve factors met
    options['pass'] = (options['RF_st'] > 1) & (options['RF_sk'] > 1) & (options['RF_mat'] > 1) & (options['RF_euler'] > 1.2)
    
    # get all the staking sequences that meet the reserve factors and are to within an average of 10% of the target thicknesses
    options_pass = options[options['pass']]
    factor = (options_pass.tst/best_tst * options_pass.tsk/best_tsk)
    options_pass = options_pass[factor < 1.2].copy()

    # if no options pass, pick the one that gives the highest reserve factor
    if len(options_pass) == 0:
        factor = (options.tst/best_tst * options.tsk/best_tsk)
        options_pass = options[factor < 1.20].copy()
        options_pass = options_pass[options_pass.Nx_euler == options_pass.Nx_euler.max()].copy()

    # ensure the skin and stiffener buckling loads are to within 10% of eachother
    factor = (options_pass.Nx_st/options_pass.Nx_sk-1).abs()
    options_pass = options_pass[factor < 0.10].copy()
    options_final = options_pass[options_pass.mass == options_pass.mass.min()].copy()

    # if more than one option still remains, pick the one that gives the lowest mass.
    if len(options_final) > 1:
        return options_final[options_final.mass == options_final.mass.min()]
    else:
        return options_final