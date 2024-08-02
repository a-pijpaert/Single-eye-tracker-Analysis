def print_mae_info(df):
    for eye in df['eye'].unique():
        for axis in ['x', 'y']:
            mean_value = df.loc[(df['eye'] == eye) & (df['mae'] == axis), 'mean'].values[0]
            std_value = df.loc[(df['eye'] == eye) & (df['mae'] == axis), 'std'].values[0]
            print(f"{eye.capitalize()} MAE{axis.upper()}: {mean_value:.2f} ± {std_value:.2f}")
