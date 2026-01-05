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
        for prod in df_forecast['Product'].unique():
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]
            
            # Union of all nodes in the network (Forecasted nodes + Hubs/Suppliers in lead time file)
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(
                set(p_lt['To_Location'])
            )
            
            if not nodes: continue

            agg_demand = {n: p_fore.get(n, {'Forecast_Quantity': 0})['Forecast_Quantity'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': 0})['Local_Std'])**2 for n in nodes}
            
            children = {}
            for _, row in p_lt.iterrows():
                if row['From_Location'] not in children: children[row['From_Location']] = []
                children[row['From_Location']].append(row['To_Location'])
                
            for _ in range(15):
                changed = False
                for parent in nodes:
                    if parent in children:
                        new_d = p_fore.get(parent, {'Forecast_Quantity': 0})['Forecast_Quantity'] + sum(agg_demand.get(c, 0) for c in children[parent])
                        new_v = (p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(agg_var.get(c, 0) for c in children[parent])
                        
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

s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    for df in [df_s, df_d, df_lt]: df.columns = [c.strip() for c in df.columns]
    
    df_s['Quantity'] = clean_numeric(df_s['Quantity'])
    df_d['Forecast_Quantity'] = clean_numeric(df_d['Forecast_Quantity'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # Calculations
    stats = df_s.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    network_stats = aggregate_network_stats(df_d, stats, df_lt)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # Merge Logic: Start with network_stats to preserve Hubs
    results = pd.merge(network_stats, df_d, on=['Product', 'Location', 'Future_Forecast_Month'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    
    results = results.fillna({'Forecast_Quantity': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})
    
    # Calculate SS
    results['Safety_Stock'] = (z * np.sqrt(
        (results['LT_Mean']/30)*(results['Agg_Std_Hist']**2) + (results['LT_Std']**2)*(results['Agg_Future_Demand']/30)**2
    )).round(0)

    # --- FORCE B616 TO ZERO ---
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast_Quantity']

    # Global Selectors in Sidebar
    st.sidebar.divider()
    selected_sku = st.sidebar.selectbox("Select Product", sorted(results['Product'].unique()))
    selected_loc = st.sidebar.selectbox("Select Location", sorted(results[results['Product'] == selected_sku]['Location'].unique()))

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis", "🕰️ Sales vs Forecast"])
    
    with tab1:
        plot_df = results[(results['Product']==selected_sku) & (results['Location']==selected_loc)].sort_values('Future_Forecast_Month')
        fig = go.Figure([
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Max_Corridor'], name='Max Corridor', line=dict(width=0)),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Forecast_Quantity'], name='Local Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Agg_Future_Demand'], name='Aggregated Demand', line=dict(color='blue', dash='dash'))
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        next_month = sorted(results['Future_Forecast_Month'].unique())[0]
        label_data = results[results['Future_Forecast_Month'] == next_month].set_index(['Product', 'Location']).to_dict('index')
        net = Network(height="900px", width="100%", directed=True)
        sku_lt = df_lt[df_lt['Product'] == selected_sku]
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        for n in all_nodes:
            m = label_data.get((selected_sku, n), {'Forecast_Quantity': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label_text = f"{n}\nLocal: {int(m['Forecast_Quantity'])}\nNet: {int(m['Agg_Future_Demand'])}\nSS: {int(m['Safety_Stock'])}"
            color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
            net.add_node(n, label=label_text, title=label_text, color=color, shape='box', font={'color': 'white'})
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
        net.save_graph("net.html")
        components.html(open("net.html", 'r').read(), height=950)

    with tab3:
        st.subheader("Global Inventory Plan")
        st.dataframe(results, use_container_width=True, height=1000) # Increased height to show more lines

    with tab4:
        st.subheader(f"⚖️ Efficiency Snapshot: {next_month}")
        eff_df = results[(results['Product'] == selected_sku) & (results['Future_Forecast_Month'] == next_month)].copy()
        eff_df['SS_to_Fcst_Ratio'] = (eff_df['Safety_Stock'] / eff_df['Forecast_Quantity'].replace(0, np.nan)).fillna(0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Safety Stock", f"{int(eff_df['Safety_Stock'].sum()):,}")
        m2.metric("Avg SS Ratio", f"{eff_df['SS_to_Fcst_Ratio'].mean():.2f}")
        m3.metric("B616 SS Status", "FORCED TO 0" if selected_loc == "B616" else "Calculated")
        st.plotly_chart(px.scatter(eff_df, x="Forecast_Quantity", y="Safety_Stock", size="Agg_Future_Demand", color="Location", hover_name="Location"), use_container_width=True)

    with tab5:
        st.subheader("🕰️ Historical Sales vs. Future Forecast")
        # Ensure Sales data has a 'Month' column to compare with 'Future_Forecast_Month'
        if 'Month' in df_s.columns:
            s_hist = df_s[(df_s['Product'] == selected_sku) & (df_s['Location'] == selected_loc)].groupby('Month')['Quantity'].sum().reset_index()
            f_hist = df_d[(df_d['Product'] == selected_sku) & (df_d['Location'] == selected_loc)].groupby('Future_Forecast_Month')['Forecast_Quantity'].sum().reset_index()
            f_hist.columns = ['Month', 'Forecast']
            
            comp = pd.merge(s_hist, f_hist, on='Month', how='outer').sort_values('Month').fillna(0)
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=comp['Month'], y=comp['Quantity'], name='Actual Sales (History)'))
            fig_comp.add_trace(go.Scatter(x=comp['Month'], y=comp['Forecast'], name='Forecast', line=dict(color='red')))
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(comp, use_container_width=True)
        else:
            st.info("To view history, ensure your Sales CSV has a 'Month' column that matches the Forecast format.")

else:
    st.warning("Please upload Sales, Demand, and Lead Time CSVs to proceed.")
