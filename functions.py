import plotly.express as px

def plot_top_categories(
    df,
    category_col,
    value_col=None,
    top_n=10,
    title="Top Categories",
    y_label="Count"
):
    """
    Plots a top-N bar chart for product categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    category_col : str
        Column containing category names
    value_col : str or None
        Column to aggregate (None = count)
    top_n : int
        Number of top categories
    title : str
        Chart title
    y_label : str
        Y-axis label
    """

    if value_col is None:
        data = (
            df[category_col]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        data.columns = ['category', 'value']
    else:
        data = (
            df.groupby(category_col, as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=False)
            .head(top_n)
        )
        data.rename(columns={
            category_col: 'category',
            value_col: 'value'
        }, inplace=True)

    # Clean category names
    data['category'] = data['category'].str.replace('_', ' ')

    fig = px.bar(
        data,
        x='category',
        y='value',
        title=title,
        labels={
            'category': 'Product Category',
            'value': y_label
        }
    )

    fig.update_layout(xaxis_tickangle=-45)

    return fig
