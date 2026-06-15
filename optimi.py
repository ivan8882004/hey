# --- Filter neutral ratings and run optimisation ---

ratings = ratings[(ratings < 2.75) | (ratings > 3.25)]

longs = ratings[ratings > 3].sort_values(ascending=False)
shorts = ratings[ratings < 3].sort_values()

top_long = longs.iloc[:int(0.15 * len(longs))].index
mid_long = longs.iloc[int(0.15 * len(longs)):int(0.35 * len(longs))].index

top_short = shorts.iloc[:int(0.15 * len(shorts))].index
mid_short = shorts.iloc[int(0.15 * len(shorts)):int(0.35 * len(shorts))].index

tickers_opt = ratings.index.tolist()
n_opt = len(tickers_opt)

Sigma = covariance.loc[tickers_opt, tickers_opt].values

idx = {t: i for i, t in enumerate(tickers_opt)}

w = cp.Variable(n_opt)

sign = np.where(ratings.values > 3, 1.0, -1.0)

constraints = []

constraints.append(cp.sum(cp.multiply(sign, w)) == 1.0)

for i in range(n_opt):

    if ratings.iloc[i] >= 4.5:
        constraints += [
            sign[i] * w[i] >= 0.002,
            sign[i] * w[i] <= 0.02
        ]

    elif ratings.iloc[i] >= 3.5:
        constraints += [
            sign[i] * w[i] >= 0.001,
            sign[i] * w[i] <= 0.02
        ]

    elif ratings.iloc[i] <= 1.5:
        constraints += [
            sign[i] * w[i] >= 0.002,
            sign[i] * w[i] <= 0.02
        ]

    else:
        constraints += [
            sign[i] * w[i] >= 0.001,
            sign[i] * w[i] <= 0.02
        ]

top_idx = [idx[t] for t in top_long] + [idx[t] for t in top_short]
mid_idx = [idx[t] for t in mid_long] + [idx[t] for t in mid_short]

other_idx = list(
    set(range(n_opt)) - set(top_idx) - set(mid_idx)
)

print(f'Tier sizes — top: {len(top_idx)}, mid: {len(mid_idx)}, other: {len(other_idx)} (total: {n_opt})')

constraints += [

    cp.sum(cp.multiply(sign[top_idx], w[top_idx])) >= 0.45,
    cp.sum(cp.multiply(sign[top_idx], w[top_idx])) <= 0.55,

    cp.sum(cp.multiply(sign[mid_idx], w[mid_idx])) >= 0.25,
    cp.sum(cp.multiply(sign[mid_idx], w[mid_idx])) <= 0.35,

    cp.sum(cp.multiply(sign[other_idx], w[other_idx])) >= 0.15,
    cp.sum(cp.multiply(sign[other_idx], w[other_idx])) <= 0.25,

]

objective = cp.Minimize(cp.quad_form(w, Sigma))

problem = cp.Problem(objective, constraints)

problem.solve()

weights = pd.Series(w.value, index=tickers_opt)
signed_weights = pd.Series(sign * w.value, index=tickers_opt)

# --- Quick summary ---
print(f'\nStatus: {problem.status}')
print(f'Portfolio variance (annualised): {problem.value:.6f}')
print(f'Portfolio vol (annualised):      {np.sqrt(problem.value)*100:.2f}%')
print(f'Positions: {len(weights)} (after filtering neutral ratings)')
print(f'Long:  {(weights > 0).sum():>4}  gross={weights[weights > 0].sum():.3f}')
print(f'Short: {(weights < 0).sum():>4}  gross={abs(weights[weights < 0].sum()):.3f}')
print(f'Net signed exposure: {signed_weights.sum():.3f}  (constraint: =1.0)')
