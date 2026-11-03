import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column: BMI > 25 is overweight.
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (bmi > 25).astype(int)

# Normalize cholesterol and glucose: 0 = good, 1 = bad.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # Convert categorical features to long format.
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
        var_name='variable',
        value_name='value',
    )

    # Count observations by cardio status, feature, and value.
    df_cat = (
        df_cat.groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Plot categorical counts in separate panels for cardio=0 and cardio=1.
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
    )
    fig = g.fig

    # Do not modify the next two lines.
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # Filter invalid blood pressures and extreme heights/weights.
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi'])
        & (df['height'] >= df['height'].quantile(0.025))
        & (df['height'] <= df['height'].quantile(0.975))
        & (df['weight'] >= df['weight'].quantile(0.025))
        & (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Correlation matrix and upper-triangle mask.
    corr = df_heat.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Draw heatmap.
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        linewidths=0.5,
        center=0,
        cmap='coolwarm',
        ax=ax,
    )

    # Do not modify the next two lines.
    fig.savefig('heatmap.png')
    return fig
