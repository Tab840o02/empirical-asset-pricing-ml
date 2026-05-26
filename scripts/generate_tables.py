"""Generate LaTeX table fragments for Phase 7 report."""
import pandas as pd
import os

os.makedirs('report/tables', exist_ok=True)

MODEL_DISPLAY = {
    'ols3':    r'\texttt{ols3}',
    'ols_all': r'\texttt{ols\_all}',
    'glm':     r'\texttt{glm}',
    'pcr':     r'\texttt{pcr}',
    'pls':     r'\texttt{pls}',
    'enet':    r'\texttt{enet}',
    'rf':      r'\texttt{rf}',
    'gbrt':    r'\texttt{gbrt}',
    'nn1':     r'\texttt{nn1}',
    'nn2':     r'\texttt{nn2}',
    'nn3':     r'\texttt{nn3}',
    'nn4':     r'\texttt{nn4}',
    'nn5':     r'\texttt{nn5}',
}

def fmt_model(s):
    return MODEL_DISPLAY.get(str(s), str(s).replace('_', r'\_'))

def add_placement(latex_str):
    return latex_str.replace('\\begin{table}\n', '\\begin{table}[H]\n\\centering\n')


# OOS R2 table (Phase 4)
df = pd.read_csv('data/processed/eval_oos_r2_latest.csv', index_col=0)
df = df.sort_values('oos_r2', ascending=False)
df.index = df.index.map(fmt_model)
df.index.name = 'Model'
df.columns = [r'OOS $R^2$']
df[r'OOS $R^2$'] = df[r'OOS $R^2$'].map('{:.4f}'.format)
lat = add_placement(df.to_latex(
    caption=r'Phase~4 pooled OOS $R^2$ (1987--2016). Benchmark: expanding-window mean excess return.',
    label='tab:oosr2', escape=False, column_format='lc'))
open('report/tables/tab_oosr2.tex', 'w', encoding='utf-8').write(lat)
print('tab_oosr2.tex written')

# IC Stats
df = pd.read_csv('data/processed/eval_ic_stats_latest.csv')
df = df.sort_values('mean_ic', ascending=False)
df.columns = ['Model', 'Mean IC', 'Std IC', 'ICIR']
df['Model'] = df['Model'].map(fmt_model)
for c in ['Mean IC', 'Std IC', 'ICIR']:
    df[c] = df[c].map('{:.4f}'.format)
lat = add_placement(df.to_latex(
    index=False,
    caption=r'Phase~4 monthly Spearman rank IC statistics (1987--2016, min 5 stocks/month).',
    label='tab:ic', escape=False, column_format='lccc'))
open('report/tables/tab_ic.tex', 'w', encoding='utf-8').write(lat)
print('tab_ic.tex written')

# Portfolio Performance
df = pd.read_csv('data/processed/eval_portfolio_perf_latest.csv')
df = df.sort_values('sharpe', ascending=False)
df = df[['model', 'annual_ret', 'annual_vol', 'sharpe', 'alpha_annual', 't_alpha', 'p_alpha']]
df['model'] = df['model'].map(fmt_model)
df.columns = ['Model', 'Ann. Ret', 'Ann. Vol', 'Sharpe', r'FF5 $\alpha$', r'$t_\alpha$', r'$p_\alpha$']
for c in ['Ann. Ret', 'Ann. Vol', 'Sharpe', r'FF5 $\alpha$']:
    df[c] = df[c].map('{:.3f}'.format)
df[r'$t_\alpha$'] = df[r'$t_\alpha$'].map('{:.2f}'.format)
df[r'$p_\alpha$'] = df[r'$p_\alpha$'].map('{:.4f}'.format)
lat = add_placement(df.to_latex(
    index=False,
    caption=r'Phase~4 long-short decile portfolio performance (1987--2016). FF5 alpha via HAC Newey-West (12 lags).',
    label='tab:portfolio', escape=False, column_format='lcccccc'))
open('report/tables/tab_portfolio.tex', 'w', encoding='utf-8').write(lat)
print('tab_portfolio.tex written')

# TC Summary (Amihud)
df = pd.read_csv('data/processed/eval_tc_summary.csv')
df_am = df[df['tc_method'] == 'amihud_calibrated'][
    ['model', 'annual_ret_gross', 'sharpe_gross', 'annual_ret_net', 'sharpe_net', 'mean_turnover', 'sharpe_delta']
].copy()
df_am = df_am.sort_values('sharpe_gross', ascending=False)
df_am['model'] = df_am['model'].map(fmt_model)
df_am.columns = ['Model', 'Gross Ret', 'Gross Sharpe', 'Net Ret', 'Net Sharpe', 'Turnover', r'$\Delta$Sharpe']
for c in ['Gross Ret', 'Gross Sharpe', 'Net Ret', 'Net Sharpe', 'Turnover', r'$\Delta$Sharpe']:
    df_am[c] = df_am[c].map('{:.3f}'.format)
lat = add_placement(df_am.to_latex(
    index=False,
    caption=r'Extension~2: Transaction-cost-adjusted performance (Amihud-calibrated spreads, 1987--2016). Rank ordering fully preserved post-TC.',
    label='tab:tc', escape=False, column_format='lcccccc'))
open('report/tables/tab_tc_summary.tex', 'w', encoding='utf-8').write(lat)
print('tab_tc_summary.tex written')

# Parsimony Delta R2
df = pd.read_csv('data/processed/eval_parsimony_oos_r2.csv', index_col=0)
df = df[['phase4_r2', 'phase5c_r2', 'delta_r2']]
df.columns = [r'Full (94 feat.)', r'Parsimonious (15 feat.)', r'$\Delta R^2$']
for c in df.columns:
    df[c] = df[c].map('{:.4f}'.format)
df.index = df.index.map(fmt_model)
df.index.name = 'Model'
lat = add_placement(df.to_latex(
    caption=r'Extension~3: OOS $R^2$ comparison: full (94 features) vs.\ parsimonious (15 features), same test window 1987--2016.',
    label='tab:parsimony', escape=False, column_format='lccc'))
open('report/tables/tab_parsimony_delta.tex', 'w', encoding='utf-8').write(lat)
print('tab_parsimony_delta.tex written')

# Parsimony Portfolio (Audited)
df = pd.read_csv('data/processed/eval_parsimony_portfolio_perf_audited.csv')
df = df.sort_values('sharpe', ascending=False)
df = df[['model', 'annual_ret', 'annual_vol', 'sharpe', 'alpha_annual', 't_alpha', 'p_alpha', 'n_months']]
df['model'] = df['model'].map(fmt_model)
df.columns = ['Model', 'Ann. Ret', 'Ann. Vol', 'Sharpe', r'FF5 $\alpha$', r'$t_\alpha$', r'$p_\alpha$', 'Months']
for c in ['Ann. Ret', 'Ann. Vol', 'Sharpe', r'FF5 $\alpha$']:
    df[c] = df[c].map('{:.3f}'.format)
df[r'$t_\alpha$'] = df[r'$t_\alpha$'].map('{:.2f}'.format)
df[r'$p_\alpha$'] = df[r'$p_\alpha$'].map('{:.4f}'.format)
lat = add_placement(df.to_latex(
    index=False,
    caption=r'Extension~3: Parsimonious model L/S portfolio performance (audited). NN4: 312/360 non-degenerate months; NN5: 144/360.',
    label='tab:parsimony_port', escape=False, column_format='lccccccr'))
open('report/tables/tab_parsimony_portfolio.tex', 'w', encoding='utf-8').write(lat)
print('tab_parsimony_portfolio.tex written')

# Post-2020 OOS R2 by sub-period
df = pd.read_csv('data/processed/eval_ext_oos_r2.csv')
pivot = df.pivot(index='model', columns='period', values='oos_r2')
col_order = ['Pre-COVID (2017-2019)', 'COVID (2020)', 'Reflation (2021)',
             'Rate hikes (2022)', 'Post-norm (2023+)', 'Full ext (2017+)']
pivot = pivot[[c for c in col_order if c in pivot.columns]]
pivot = pivot.round(4)
pivot.index = pivot.index.map(fmt_model)
pivot.index.name = 'Model'
pivot.columns.name = ''
lat = add_placement(pivot.to_latex(
    caption=r'Extension~1: Post-2020 OOS $R^2$ by sub-period (Phase~4 models applied forward to 2017--2024).',
    label='tab:ext_oosr2', escape=False, na_rep=r'---',
    column_format='l' + 'c' * len(pivot.columns)))
open('report/tables/tab_ext_oosr2.tex', 'w', encoding='utf-8').write(lat)
print('tab_ext_oosr2.tex written')

print('\nAll tables generated successfully!')

