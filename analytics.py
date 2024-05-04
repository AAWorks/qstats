import pandas as pd
import seaborn as sns
import streamlit as st
import altair as alt
import PIL.Image
from stqdm import stqdm

from parsing import parse_asset_data

if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image

class StreamlitReport:
    def __init__(self, assets, depth, requested_methods):
        self._df = parse_asset_data(st.warning, depth, assets)
        self._assets = assets
        self._methods = [k for k, v in requested_methods.items() if k[v]]

        self._all_methods = {
            "df_snapshot": (self.get_df, st.dataframe), # func, viz_method
            "simple_plot": (self.simple_plot, st.altair_chart),
            "price_distro": (self.price_distributions, st.altair_chart),
            "simple_sma": (self.simple_sma_graph, st.altair_chart),
            "asset_vol": (self.asset_volatility, st.altair_chart),
            "price_corr_map": (self.price_correlation_map, st.altair_chart),
            "price_corr_viz": (self.price_correlation_graphs, st.pyplot),
            "sma_corr_map": (self.sma_correlation_map, st.altair_chart),
            "monthly_returns": (self.monthly_returns, st.altair_chart)
        }

        self._metric_names = {
            "df_snapshot": "DataFrame Snapshot",
            "simple_plot": "Simple Mutli-Asset Graph",
            "price_distro": "Asset Price Distributions",
            "simple_sma": "Price & 30-Day SMA",
            "asset_vol": "Asset Volatilities",
            "price_corr_map": "Price Correlation Matrix",
            "price_corr_viz": "Price Correlation Visualizer",
            "sma_corr_map": "SMA Correlation Matrix",
            "monthly_returns": "Monthly Returns Correlation Matrix"
        }

        sns.set_style("whitegrid")
    
    @property
    def df(self):
        return self._df

    def get_df(self):
        return self.df
    
    @property
    def all_methods(self):
        return self._all_methods.keys()
    
    def simple_plot(self):
        df_melted = self.df.melt('Date', var_name='Ticker', value_name='Price')
        chart = alt.Chart(df_melted).mark_line().encode(
            x='Date:T',  # T: Temporal field for time data
            y='Price:Q',  # Q: Quantitative field for numerical data
            color='Ticker:N',  # N: Nominal field for categorical data
            tooltip=['Date:T', 'Price:Q', 'Ticker:N']
        ).properties(
            width=800,
            height=400,
            title='Time Series of Asset Prices'
        ).interactive()
        
        return chart

    def price_distributions(self, streamlit=False):
        df_melted = self.df.melt('Date', var_name='Ticker', value_name='Price')

        base_chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('Price:Q', 
                    axis=alt.Axis(title='Price'),
                    scale=alt.Scale(zero=False),
                    bin=alt.Bin(maxbins=20)),
            y=alt.Y('count():Q', 
                    axis=alt.Axis(title='Frequency')),
            color=alt.Color('Ticker:N', legend=alt.Legend(title="Ticker"))
        ).properties(
            width=260,
            height=200
        )

        concat_chart = alt.ConcatChart(
            concat=[
                base_chart.transform_filter(alt.datum.Ticker == ticker).properties(title=ticker)
                for ticker in sorted(df_melted['Ticker'].unique())
            ],
            columns=3 
        ).configure_title(
            fontSize=12,
            font='Courier',
            anchor='middle',
            color='gray',
            align='left'
        ).resolve_axis(
            x='independent',
            y='independent'
        ).resolve_scale(
            x='independent',
            y='independent'
        )
        
        return concat_chart

    def simple_sma_graph(self):
        df = self.df.drop("Date", axis=1)
        rolling_mean = df.rolling(window=30).mean()

        df_reset = df.reset_index().melt('index', var_name='Ticker', value_name='Price')
        rolling_mean_reset = rolling_mean.reset_index().melt('index', var_name='Ticker', value_name='RollingMean')

        df_full = pd.merge(df_reset, rolling_mean_reset, on=['index', 'Ticker'], how='left')

        base = alt.Chart(df_full).encode(
            x='index:T',  # Encoding time on the X-axis
            color='Ticker:N'  # Color by asset ticker
        )
        line = base.mark_line().encode(
            y=alt.Y('Price:Q', axis=alt.Axis(title='Price'))
        )
        rolling_line = base.mark_line(strokeDash=[5, 5]).encode(
            y=alt.Y('RollingMean:Q', axis=alt.Axis(title='30-Day Rolling Mean'))
        )
        chart = alt.layer(line, rolling_line).properties(
            width=700,
            height=400,
            title='Time Series of Asset Prices and 30-Day Rolling Mean'
        )

        return chart
    
    def asset_volatility(self): #using z-score for volatility
        df = self.df.drop("Date", axis=1)
        normalized_df = (df - df.mean()) / df.std()
        df_melted = normalized_df.reset_index(drop=True).melt(var_name='Asset', value_name='Normalized Price')
        chart = alt.Chart(df_melted).mark_boxplot().encode(
            x=alt.X('Asset:N', title='Asset', axis=alt.Axis(labelAngle=-45)),  # Asset names on the x-axis with rotated labels
            y=alt.Y('Normalized Price:Q', title='Normalized Price'),  # Normalized prices on the y-axis
            color='Asset:N'  # Color the boxplots by asset for clearer distinction
        ).properties(
            width=400,
            height=300,
            title='Normalized Distribution of Asset Prices (Boxplots)'
        )
        
        return chart
    
    def price_correlation_map(self):
        df = self.df.drop("Date", axis=1)
        corr_matrix = df.corr()
        corr_df = corr_matrix.reset_index().melt('index', var_name='Column', value_name='Correlation')
        corr_df.rename(columns={'index': 'Row'}, inplace=True)
        
        heatmap = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X('Row:N', title=None),
            y=alt.Y('Column:N', title=None),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1)), legend=alt.Legend(title="Correlation"))
        )

        text = heatmap.mark_text(baseline='middle').encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                "datum.Correlation > 0.5 || datum.Correlation < -0.5",
                alt.value('white'),
                alt.value('black')
            )
        )

        final_chart = (heatmap + text).properties(
            width=300,
            height=100 * len(df.columns),
            title='Correlation Heatmap of Asset Prices'
        )

        return final_chart
    
    def price_correlation_graphs(self):
        return sns.pairplot(self.df)
    
    def sma_correlation_map(self):
        df = self.df.drop("Date", axis=1)
        rolling_mean_30 = df.rolling(window=30).mean()
        corr_matrix = rolling_mean_30.corr()
        corr_df = corr_matrix.stack().reset_index()
        corr_df.columns = ['Variable1', 'Variable2', 'Correlation']

        heatmap = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X('Variable1:N', title='Ticker', sort=list(df.columns)),
            y=alt.Y('Variable2:N', title='Ticker', sort=list(df.columns)),
            color=alt.Color('Correlation:Q', scale=alt.Scale(domain=[-1, 1], scheme='redblue'))
        )

        text = heatmap.mark_text().encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                "datum.Correlation > 0.5 || datum.Correlation < -0.5",
                alt.value('white'),
                alt.value('black')
            )
        )
        final_chart = (heatmap + text).properties(
            width=300,
            height=100 * len(df.columns),
            title='Correlation Heatmap of 30-Day Moving Averages of Asset Prices'
        )
        
        return final_chart
    
    def monthly_returns(self):
        df = self.df.set_index("Date")
        monthly_returns = df.resample('M').ffill().pct_change()

        correlation_matrix = monthly_returns.corr()
        corr_df = correlation_matrix.stack().reset_index()
        corr_df.columns = ['Variable1', 'Variable2', 'Correlation']

        heatmap = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X('Variable1:N', title='Asset', sort=list(monthly_returns.columns)),
            y=alt.Y('Variable2:N', title='Asset', sort=list(monthly_returns.columns)),
            color=alt.Color('Correlation:Q', scale=alt.Scale(domain=[-1, 1], scheme='redblue'))
        )

        text = heatmap.mark_text().encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                "datum.Correlation > 0.5 || datum.Correlation < -0.5",
                alt.value('white'),
                alt.value('black')
            )
        )

        final_chart = (heatmap + text).properties(
            width=300,
            height=100 * len(df.columns),
            title='Correlation Heatmap of Monthly Returns'
        )

        return final_chart
    
    def generate_report(self, requested_metrics: dict, expanded=True):
        metric_keys = tuple(requested_metrics.keys())
        plots = []
        for i in stqdm(range(len(metric_keys))):
            metric = metric_keys[i]
            fig, viz_func = self._all_methods[metric]
            plots.append((metric, fig(), viz_func))
                
        
        for metric, fig, viz_func in plots:
            with st.expander(self._metric_names[metric], expanded=expanded):
                viz_func(fig, use_container_width=True)
            
    def __str__(self):
        return f"Report on {', '.join(self._assets)} | Methods: {', '.join(self._methods)}"