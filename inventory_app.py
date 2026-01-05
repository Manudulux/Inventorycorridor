import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer", layout="wide")
st.title("📊 Multi-Echelon Network Inventory Optimizer")

# --- Helper Functions ---
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', '').str.replace('-', '0').str.strip(), 
        errors='coerce'
    ).fillna(0)

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    results = []
    months = df_forecast['Future_Forecast_Month'].unique()
    
    for month in months:
        df_month = df_forecast[df_forecast['Future_Forecast_Month'] == month]
        for prod in df_month['Product'].unique():
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]
            
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(set(p_lt['From_Location'])).union(set(p_lt['To_Location']))
            
            agg_demand = {n: p_fore.get(n, {'Forecast_Quantity': 0})['Forecast_Quantity'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': 0})['Local_Std'])**2 for n in nodes}
            
            children = {}
            for _, row in p_lt.iterrows():
                if row['From_Location'] not in children: children[row['From_Location']] = []
                children[row['From_Location']].append(row['To_Location'])
                
            for _ in range(10):
                changed = False
                for parent in nodes:
                    if parent in children:
                        new_d = p_fore.get(parent, {'Forecast_Quantity': 0})['Forecast_Quantity'] + sum(agg_demand[c] for c in children[parent])
                        new_v = (p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(agg_var[c] for c in children[parent])
                        if abs(new_d - agg_demand[parent]) > 0.01:
                            agg_demand[parent], agg_var[parent] = new_d, new_v
                            changed = True
                if not changed: break
                
            for n in nodes:
                results.append({
                    'Product': prod, 
                    'Location': n, 
                    'Future_Forecast_Month': month,
                    'Agg_Future_Demand': agg_demand[n], 
                    'Agg_Std_Hist': np.sqrt(agg_var[n])
                })
    return pd.DataFrame(results)

# --- Sidebar Parameter & File Upload ---
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 80.0, 99.9, 95.0)/100
z = norm.ppf(service_level)

s_file = st.sidebar.file_uploader("1. Sales Data", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data", type="csv")

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    for df in [df_s, df_d, df_lt]: df.columns = [c.strip() for c in df.columns]
    
    df_s['Quantity'] = clean_numeric(df_s['Quantity'])
    df_d['Forecast_Quantity'] = clean_numeric(df_d['Forecast_Quantity'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    stats = df_s.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    network_stats = aggregate_network_stats(df_d, stats, df_lt)

    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(df_d, network_stats, on=['Product', 'Location', 'Future_Forecast_Month'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left').fillna({'Agg_Std_Hist':0, 'LT_Mean':7, 'LT_Std':2, 'Agg_Future_Demand':0})
    
    results['Safety_Stock'] = (z * np.sqrt((results['LT_Mean']/30)*(results['Agg_Std_Hist']**2) + (results['LT_Std']**2)*(results['Agg_Future_Demand']/30)**2)).round(0)
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast_Quantity']

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis"])
    
    with tab1:
        sku = st.selectbox("Product", results['Product'].unique())
        loc = st.selectbox("Location", results[results['Product']==sku]['Location'].unique())
        plot_df = results[(results['Product']==sku) & (results['Location']==loc)].sort_values('Future_Forecast_Month')
        fig = go.Figure([
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Max_Corridor'], name='Max Corridor', line=dict(width=0)),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Forecast_Quantity'], name='Local Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Agg_Future_Demand'], name='Aggregated Network Demand', line=dict(color='blue', dash='dash'))
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Labels show: Location | Local Forecast | Network Demand | Proposed SS (Next Month Data)")
        # Get data for the earliest month to show as labels
        next_month = sorted(results['Future_Forecast_Month'].unique())[0]
        label_data = results[results['Future_Forecast_Month'] == next_month].set_index(['Product', 'Location']).to_dict('index')

        net = Network(height="900px", width="100%", directed=True)
        sku_lt = df_lt[df_lt['Product'] == sku]
        
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        
        for n in all_nodes:
            # Fetch metrics for the node
            m = label_data.get((sku, n), {'Forecast_Quantity': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label_text = (f"{n}\n"
                          f"Local: {int(m['Forecast_Quantity'])}\n"
                          f"Net: {int(m['Agg_Future_Demand'])}\n"
                          f"SS: {int(m['Safety_Stock'])}")
            
            color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
            net.add_node(n, label=label_text, title=label_text, color=color, shape='box', font={'color': 'white'})

        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
            
        net.save_graph("net.html")
        components.html(open("net.html", 'r').read(), height=950)

    with tab3:
        st.subheader("Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        with col1: f_prod = st.multiselect("Filter Product", results['Product'].unique(), key="f1")
        with col2: f_loc = st.multiselect("Filter Location", results['Location'].unique(), key="f2")
        with col3: f_month = st.multiselect("Filter Month", sorted(results['Future_Forecast_Month'].unique()), key="f3")

        filtered_df = results.copy()
        if f_prod: filtered_df = filtered_df[filtered_df['Product'].isin(f_prod)]
        if f_loc: filtered_df = filtered_df[filtered_df['Location'].isin(f_loc)]
        if f_month: filtered_df = filtered_df[filtered_df['Future_Forecast_Month'].isin(f_month)]

        st.dataframe(filtered_df[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity', 'Agg_Future_Demand', 'Safety_Stock', 'Max_Corridor']], use_container_width=True)

    with tab4:
        st.subheader("⚖️ Network Efficiency & Risk Dashboard")
        st.markdown("This analysis identifies locations where Safety Stock is high relative to Local Forecast, often caused by upstream lead time variability.")
        
        # Calculate SS as % of Forecast
        eff_df = results[results['Product'] == sku].copy()
        eff_df['SS_as_Percent_of_Forecast'] = (eff_df['Safety_Stock'] / eff_df['Forecast_Quantity'].replace(0, np.nan) * 100).fillna(0)
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Scatter plot: Forecast vs Safety Stock
            fig_eff = px.scatter(eff_df, x="Forecast_Quantity", y="Safety_Stock", color="Location",
                                 size="Agg_Future_Demand", hover_data=["Future_Forecast_Month"],
                                 title=f"Safety Stock vs Local Forecast for {sku}")
            st.plotly_chart(fig_eff, use_container_width=True)
            
        with c2:
            # Top "Stock Heavy" Locations
            avg_ss_pct = eff_df.groupby('Location')['SS_as_Percent_of_Forecast'].mean().sort_values(ascending=False).reset_index()
            st.write("**Top 5 Nodes by SS-to-Forecast Ratio**")
            st.table(avg_ss_pct.head(5).rename(columns={'SS_as_Percent_of_Forecast': 'Avg SS %'}))
            
        st.divider()
        st.markdown("**Insight:** Nodes with high SS but low local forecast are likely serving as 'Network Buffers' for downstream locations.")

        # Download
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Optimization Data", csv, "inventory_optimization_full.csv", "text/csv")
