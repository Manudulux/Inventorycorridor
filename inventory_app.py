import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm
import tempfile
import os
import io

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
            p_stats_df = df_stats[df_stats['Product'] == prod]
            p_stats = p_stats_df.groupby('Location').mean(numeric_only=True).to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]
            
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(set(p_lt['To_Location']))
            
            if not nodes: continue

            agg_demand = {n: p_fore.get(n, {'Forecast_Quantity': 0})['Forecast_Quantity'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': agg_demand[n] * 0.2})['Local_Std'])**2 for n in nodes}
            
            children = {}
            for _, row in p_lt.iterrows():
                if row['From_Location'] not in children: children[row['From_Location']] = []
                children[row['From_Location']].append(row['To_Location'])
                
            for _ in range(15):
                changed = False
                for parent in nodes:
                    if parent in children:
                        parent_local_std = p_stats.get(parent, {'Local_Std': agg_demand[parent] * 0.2})['Local_Std']
                        new_d = p_fore.get(parent, {'Forecast_Quantity': 0})['Forecast_Quantity'] + sum(agg_demand.get(c, 0) for c in children[parent])
                        new_v = (parent_local_std)**2 + sum(agg_var.get(c, 0) for c in children[parent])
                        
                        if abs(new_d - agg_demand[parent]) > 0.01:
                            agg_demand[parent], agg_var[parent] = new_d, new_v
                            changed = True
                if not changed: break
                
            for n in nodes:
                results.append({
                    'Product': prod, 'Location': n, 'Future_Forecast_Month': month,
                    'Agg_Future_Demand': agg_demand[n], 'Agg_Std_Hist': np.sqrt(agg_var[n])
                })
    return pd.DataFrame(results)

# --- Top of Page Parameters ---
service_level = st.slider("Target Service Level (%)", 90.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

# --- Sidebar File Upload ---
st.sidebar.header("📁 Data Upload")
s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    for df in [df_s, df_d, df_lt]: df.columns = [c.strip() for c in df.columns]
    
    # Data Cleaning & Date Handling
    df_s['Quantity'] = clean_numeric(df_s['Quantity'])
    if 'Historical_Forecast' in df_s.columns:
        df_s['Historical_Forecast'] = clean_numeric(df_s['Historical_Forecast'])
    if 'Month/Year' in df_s.columns:
        df_s['Date_Sort'] = pd.to_datetime(df_s['Month/Year'], errors='coerce')
    
    df_d['Forecast_Quantity'] = clean_numeric(df_d['Forecast_Quantity'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # Calculation of Historical Stats for Variability
    stats = df_s.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)
    stats['CV'] = (stats['Local_Std'] / stats['Local_Mean'].replace(0, np.nan)).fillna(0)

    # Core Logic
    network_stats = aggregate_network_stats(df_d, stats, df_lt)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean(numeric_only=True).reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d, on=['Product', 'Location', 'Future_Forecast_Month'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    
    results = results.fillna({'Forecast_Quantity': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})
    
    # Safety Stock Formula: SS = z * sqrt( (LT_avg * Var_demand) + (Var_LT * Demand_avg^2) )
    results['Safety_Stock'] = (z * np.sqrt(
        (results['LT_Mean']/30) * (results['Agg_Std_Hist']**2) + 
        (results['LT_Std']**2) * (results['Agg_Future_Demand']/30)**2
    )).round(0)
    
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast_Quantity']

    # --- Analysis UI ---
    sku = st.selectbox("Select Product", sorted(results['Product'].unique()))
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis", "🕰️ Forecast Accuracy & Variability"
    ])
    
    with tab1:
        loc = st.selectbox("Select Location", sorted(results[results['Product']==sku]['Location'].unique()))
        plot_df = results[(results['Product']==sku) & (results['Location']==loc)].sort_values('Future_Forecast_Month')
        
        fig = go.Figure([
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Fcst)', line=dict(width=0)),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Forecast_Quantity'], name='Local Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Agg_Future_Demand'], name='Aggregated Network Demand', line=dict(color='blue', dash='dash'))
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Network Map: {sku}")
        next_month = sorted(results['Future_Forecast_Month'].unique())[0]
        label_data = results[(results['Future_Forecast_Month'] == next_month) & (results['Product'] == sku)].set_index('Location').to_dict('index')

        net = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
        sku_lt = df_lt[df_lt['Product'] == sku]
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        
        for n in all_nodes:
            m = label_data.get(n, {'Forecast_Quantity': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label_text = f"{n}\nNet Demand: {int(m['Agg_Future_Demand'])}\nSS: {int(m['Safety_Stock'])}"
            is_hub = n in sku_lt['From_Location'].values and n not in sku_lt['To_Location'].values
            color = '#31333F' if is_hub else '#ff4b4b'
            size = 15 + (np.log1p(m['Agg_Future_Demand']) * 6)
            net.add_node(n, label=label_text, title=label_text, color=color, shape='dot', size=size)

        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d", color="#888888")
        
        net.toggle_physics(True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            os.unlink(tmp.name)
        components.html(html_content, height=700)

    with tab3:
        st.subheader("Global Inventory Plan")
        
        # Prepare data for export
        export_df = results[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity', 'Agg_Future_Demand', 'Safety_Stock', 'Max_Corridor']].copy()
        
        # Excel Download Logic
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Safety_Stock_Plan')
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Full Plan as Excel",
            data=processed_data,
            file_name=f"Inventory_Plan_{sku}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.dataframe(export_df, use_container_width=True)

    with tab4:
        st.subheader("Efficiency Snapshots")
        eff_df = results[(results['Product'] == sku) & (results['Future_Forecast_Month'] == next_month)].copy()
        fig_eff = px.scatter(eff_df, x="Forecast_Quantity", y="Safety_Stock", size="Agg_Future_Demand", color="Location", 
                             hover_name="Location", title="Strategic Buffering Analysis")
        st.plotly_chart(fig_eff, use_container_width=True)

    with tab5:
        st.subheader("🕰️ Past Data Analysis")
        hist_df = df_s[(df_s['Product'] == sku) & (df_s['Location'] == loc)].sort_values('Date_Sort')
        
        if not hist_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=hist_df['Month/Year'], y=hist_df['Quantity'], name='Actual Sales', line=dict(color='green', width=3)))
                if 'Historical_Forecast' in hist_df.columns:
                    fig_hist.add_trace(go.Scatter(x=hist_df['Month/Year'], y=hist_df['Historical_Forecast'], name='Historical Forecast', line=dict(color='orange', dash='dot')))
                fig_hist.update_layout(title=f"Historical Trend for {sku} at {loc}", xaxis_title="Month/Year", yaxis_title="Units")
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                ls_filter = stats[(stats['Product'] == sku) & (stats['Location'] == loc)]
                if not ls_filter.empty:
                    ls = ls_filter.iloc[0]
                    st.metric("Avg Monthly Sales", f"{ls['Local_Mean']:.1f}")
                    st.metric("Variability (CV)", f"{ls['CV']:.2f}")
                    if 'Historical_Forecast' in hist_df.columns and hist_df['Quantity'].sum() > 0:
                        # MAPE Calculation
                        abs_error = np.abs(hist_df['Quantity'] - hist_df['Historical_Forecast'])
                        mape = (abs_error / hist_df['Quantity'].replace(0, np.nan)).mean() * 100
                        st.metric("Forecast Accuracy", f"{max(0, 100-mape):.1f}%")
                        
                        # Bias Calculation
                        bias = (hist_df['Historical_Forecast'].sum() / hist_df['Quantity'].sum() - 1) * 100
                        st.metric("Forecast Bias", f"{bias:+.1f}%")
        else:
            st.warning("No historical data found for this Product/Location combination.")
        
        st.divider()
        st.markdown("**Variability (CV) Heatmap across Locations**")
        st.plotly_chart(px.bar(stats[stats['Product']==sku].sort_values('CV', ascending=False), x='Location', y='CV', color='CV', 
                               color_continuous_scale='Reds', title="Demand Instability Comparison"), use_container_width=True)

else:
    st.info("Please upload historical sales, future forecast, and network route files in the sidebar to begin.")
