import csv
import streamlit as st 
import pandas as pd
from pandas_datareader import data as web
import altair as alt
from typing import List
import yfinance as yf
from bs4 import BeautifulSoup as bs
import requests
import numpy as np


# Setting up page configurations
st.set_page_config(page_title="Crypto Analyzer | Beta", 
                    page_icon="./images/icon_navigador.png", 
                    layout="wide", 
                    initial_sidebar_state="auto", 
                    menu_items=None)


@st.cache
def get_symbols(url: str) -> List:
    """Scrapes the top 25 crypto symbols
        from yahoo finance

    Args:
        url (str): Link of the Yahoo Finance crypto symbols

    Returns:
        List: Symbols
    """
    # Parse the html
    html = requests.get(url).text
    parsed = bs(html, 'html5lib')
    # Scrape the top 25 symbols and store them in a list
    top_25 = []
    for value in parsed("a", {"data-test":"quoteLink"}):
        top_25.append(value.text.strip())
    # Return the list of symbols
    return top_25
    
    
def get_data(symbols: List) -> pd.DataFrame:
    """Collects the data using the Yahoo Finance API 

    Args:
        symbols (List): Symbols scraped from Yahoo Finance

    Returns:
        pd.DataFrame: Symbols as columns, Close Price as values, Date as index.
    """
    tickers = yf.Tickers(symbols)
    raw = tickers.history(period="max")['Close']
    return raw
    
 
def transform_to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms Close Price data to a Log Returns data of 
    the crypto symbols.

    Args:
        df (pd.DataFrame): Close Price data of the symbols

    Returns:
        pd.DataFrame: Log Returns of the symbols
    """
    df = df.copy()
    df_return = df.apply(lambda x: np.log(x.dropna() / x.dropna().shift(1)))
    return df_return
    
    
def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a DataFrame containing the Pearson 
    correlation coefficient in the last column.

    Args:
        df (pd.DataFrame): Close Price data of the crypto symbols

    Returns:
        pd.DataFrame: The first two columns contains the crossing symbols as values,
        the third (last) column contains the correlation coefficient as values.
    """
    df_returns = transform_to_returns(df)
    corr = df_returns.dropna().corr().stack().reset_index()
    corr.rename(columns={0:'correlation'}, inplace=True)
    return corr

@st.cache
def get_market_cap_data(symbols: List) -> pd.DataFrame:
    df_market_cap = web.get_quote_yahoo(symbols)['marketCap'].to_frame()
    return df_market_cap
    
@st.cache
def convert_df(df: pd.DataFrame) -> csv:
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def run():
    """Runs the hole app
    """
    
    st.title("Crypto Analyzer App")
    
    # Scrape the tickers
    HOME_URL = 'https://finance.yahoo.com/cryptocurrencies/'
    symbols = get_symbols(url=HOME_URL)
    
    # About expander
    expander_bar = st.expander("About the App", expanded=False)
    expander_bar.markdown("* This app is intended to be used for educational purpose.\n* The data is collected from Yahoo Finance through its API. For more info: https://pypi.org/project/yfinance/.\n* Daily Close Price is used for the analysis.")
    # Allow the user to choose the cryptos to analyze
    options = st.multiselect('Select cryptos to analyze', 
                             symbols, 
                             default=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'BNB-USD'],)
    
    # Collecting data after choosing the symbols (min two symbols)
    if len(options) > 1: 
        data = get_data(options)
        # Display table of the data
        # st.dataframe(data)
        # Get the correlation of the asset's returns
        corr = corr_matrix(data)
        
        #Graphs
        ## Heatmap base
        hm = alt.Chart(corr).mark_rect().encode(
                x=alt.X('level_1:O', axis=alt.Axis(title='')),
                y=alt.Y('level_0:O', axis=alt.Axis(title='')),
                # color=alt.Color('correlation:Q', scale=alt.Scale(domain=[0,1])),
                color='correlation:Q',
                tooltip=[alt.Tooltip('correlation', format=',.2f')],
                )
        ## Configure text
        text = hm.mark_text().encode(
            text=alt.Text('correlation:Q', format=',.2f', ),
            color=alt.condition(
                alt.datum['correlation'] > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        ## Whole heatmap
        heatmap = (hm + text)
        #
        st.markdown("\n## Pearson Correlation Coefficient", )
        st.caption("Correlation of the Log Returns of the crypto assets")
        st.altair_chart(heatmap.properties(height=500), use_container_width=True)
        st.info("""* A correlation of *+1* means a perfect positive linear relationship between the selected assets.
\n* A correlation of *0* means that the assets are not related.
\n* A correlation of *-1* means a perfect negative linear relationship between the selected assets.""")
        
        ## Mean - Variance Analysis
        ### Prepering log returns data
        st.markdown("\n## Mean - Variance Analysis", )
        df_returns = transform_to_returns(data) # log returns df
        returns = df_returns.reset_index().melt("Date", var_name="symbol", value_name="returns").copy()
        summary = df_returns.agg(['mean', 'std']).T
        market_cap = get_market_cap_data(options)
        summary = summary.join(market_cap)
        summary = summary.reset_index()
        summary.rename(columns={"index":"symbol"}, inplace=True)
        
        selector = alt.selection_multi(empty='all', fields=['symbol'])
        domain = list(summary['symbol'].unique())
        color_scale = alt.Scale(domain=domain, scheme="viridis")

        bubble = alt.Chart(summary).mark_point(filled=True).encode(
            x=alt.X("std:Q", title="Risk"),
            y=alt.Y("mean:Q", title="Return"),
            color=alt.condition(selector,
                                'symbol',
                                alt.value('lightgray'),
                                scale=color_scale),
            size=alt.Size("marketCap", legend=None),
            tooltip=[alt.Tooltip("symbol", title="Symbol"),
                    alt.Tooltip("mean", format=",.4f", title="Mean"),
                    alt.Tooltip("std", format=",.4f", title="Standard Deviation"), 
                    alt.Tooltip("marketCap", format="$,.0f", title="Market Cap")]
        ).add_selection(selector).interactive()

        hists = alt.Chart(returns).mark_bar(opacity=0.5, thickness=100).encode(
            x=alt.X('returns:Q',bin=alt.Bin(maxbins=100)),
            y=alt.Y('count()',stack=None,title="Count of Log Returns"),
            color=alt.Color('symbol:N',
                            scale=color_scale)
        ).add_selection(selector).transform_filter(
            selector
        ).interactive()
        # bubble chart and distribution of log returns chart together
        mv_chart = alt.hconcat(bubble, hists)
        st.altair_chart(mv_chart, use_container_width=True)
        
        
                
        # Download Data
        dl1, dl2, dl3 = st.columns(3)
        with dl1:                    
            ## Close Price data
            csv_price = convert_df(data)
            st.download_button(label="Download Close Price data as CSV", 
                            data=csv_price, 
                            file_name="close_prices.csv", 
                            mime="text/csv")
        with dl2:    
            ## Close Price data and Log Returns
            symbols = list(df_returns.columns) # store the column names to be change
            for symbol in symbols:
                # change the column names to be different in order to join dataframes later
                # The 'r' prefix is added to the columns of the returns dataframe
                df_returns.rename(columns={f'{symbol}':f'r{symbol}'}, inplace=True)
            price_return_data = data.join(df_returns, how='left') 
            csv_lreturns = convert_df(price_return_data)
            st.download_button(label="Download Log Returns data as CSV", 
                            data=csv_lreturns, 
                            file_name="close_prices.csv", 
                            mime="text/csv")
        with dl3:
            ## Correlation Coefficients data
            csv_corr = convert_df(corr)
            st.download_button(label="Download Correlation Coefficients data as CSV", 
                            data=csv_corr, 
                            file_name="correlations.csv", 
                            mime="text/csv")
        
    
    

if __name__ == '__main__':
    run()