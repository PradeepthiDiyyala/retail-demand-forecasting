import streamlit as st
import pandas as pd
from pathlib import Path
from predict import forecast
from inventory import inventory_policy

st.set_page_config(page_title='Retail Demand Forecasting', page_icon='chart', layout='wide')
st.title('Retail Demand Forecasting & Inventory Planning')

df = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'sales.csv')
store = st.selectbox('Store', sorted(df.store_id.unique()))
product = st.selectbox('Product', sorted(df.product_id.unique()))
days = st.slider('Forecast horizon', 1, 30, 7)
history = df[(df.store_id == store) & (df.product_id == product)].copy()

if st.button('Generate forecast'):
    result = forecast(history, days)
    policy = inventory_policy(result.units_sold.values)
    a, b, c = st.columns(3)
    a.metric('Avg daily demand', policy['avg_daily_demand'])
    b.metric('Reorder point', policy['reorder_point'])
    c.metric('Safety stock', policy['safety_stock'])
    st.line_chart(result.set_index('date')['units_sold'])
    st.dataframe(result[['date', 'units_sold']], use_container_width=True)
