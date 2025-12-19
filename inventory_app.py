import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Supply Chain Corridor Pro", layout="wide")

st.title("🌐 Supply Chain Corridor & Network Pro")
st.markdown("Advanced Inventory Optimization with Network Visualization.")

# --- Helper Functions ---
def clean_numeric(series):
    """Cleans strings like ' 1,234 ' or ' - ' into floats."""
    return pd.to_numeric(
        series.astype(str).str.replace(',', '').str.replace('-', '0').str.strip(), 
        errors='coerce'
    ).fillna(0)

# --- Sidebar ---
st.sidebar.header("⚙️ Optimization Parameters")
# Defaulting to 99.5%
service_level = st.sidebar.slider("Service Level (%)", 80.0, 99.9, 99.5, step=0.1)
z_score = norm.ppf(service_level / 100)

st.sidebar.header("📂 Data Upload")
sales_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
demand_file = st.sidebar.file_uploader("2. Demand Data (Future)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data", type="csv")

def process_data(s_raw, d_raw, lt_raw):
    # Clean Column Names
    s_raw.columns = [c.strip() for c in s_raw.columns]
    d_raw.columns = [c.strip() for c in d_raw.columns]
    lt_raw.columns = [c.strip() for c in lt_raw.columns]

    # Clean Numeric Values
    s_raw['Quantity'] = clean_numeric(s_raw['Quantity'])
    d_raw['Forecast_Quantity'] = clean_numeric(d_raw['Forecast_Quantity'])
    lt_raw['Lead_Time_Days'] = clean_numeric(lt_raw['Lead_Time_Days'])
    lt_raw['Lead_Time_Std_Dev'] = clean_numeric(lt_raw['Lead_Time_Std_Dev'])

    # Aggregate Sales (Historical Volatility)
    s_stats = s_raw.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    s_stats.columns = ['Product', 'Location', 'Avg_Sales', 'Std_Sales']
    
    # Aggregate Future Forecast (Trend)
    d_stats = d_raw.groupby(['Product', 'Location'])['Forecast_Quantity'].mean().reset_index()
    d_stats.rename(columns={'Forecast_Quantity': 'Avg_Forecast'}, inplace=True)

    # Aggregate Lead Times (Handle multiple sources for one location)
    lt_stats = lt_raw.groupby(['Product', 'To_Loc'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    lt_stats.rename(columns={'To_Loc': 'Location', 'Lead_Time_Days': 'LT_Avg', 'Lead_Time_Std_Dev': 'LT_SD'}, inplace=True)

    # Merge
    merged = pd.merge(s_stats, d_stats, on=['Product', 'Location'], how='inner')
    merged = pd.merge(merged, lt_stats, on=['Product', 'Location'], how='left')
    
    # Fill missing LT data
    merged['LT_Avg'] = merged['LT_Avg'].fillna(merged['LT_Avg'].mean() if not merged['LT_Avg'].isna().all() else 30)
    merged['LT_SD'] = merged['LT_SD'].fillna(5)

    # SAFETY STOCK CALCULATION (Monthly normalization)
    # SS = Z * sqrt( (LT/30)*Std_Sales^2 + Avg_Sales^2*(LT_SD/30)^2 )
    lt_m = merged['LT_Avg'] / 30
    ltsd_m = merged['LT_SD'] / 30
    merged['Safety_Stock'] = z_score * np.sqrt(
        (lt_m * (merged['Std_Sales']**2)) + 
        ((merged['Avg_Sales']**2) * (ltsd_m**2))
    )
    
    merged['Min_Corridor'] = merged['Safety_Stock']
    merged['Max_Corridor'] = merged['Safety_Stock'] + merged['Avg_Forecast']
    
    return merged, s_raw, d_raw, lt_raw

# --- MAIN APP LOGIC ---
if sales_file and demand_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(sales_file), pd.read_csv(demand_file), pd.read_csv(lt_file)
    data, s_full, d_full, lt_full = process_data(df_s, df_d, df_lt)

    tab1, tab2, tab3 = st.tabs(["📉 Corridor Analysis", "🌐 Network Topology", "📋 Global Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        sku = col1.selectbox("Select SKU", options=sorted(data['Product'].unique()))
        loc = col2.selectbox("Select Location", options=sorted(data[data['Product']==sku]['Location'].unique()))
        
        row = data[(data['Product'] == sku) & (data['Location'] == loc)].iloc[0]
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Safety Stock", f"{row['Safety_Stock']:.0f}")
        m2.metric("LT Average (Days)", f"{row['LT_Avg']:.1f}")
        m3.metric("Sales Volatility (SD)", f"{row['Std_Sales']:.1f}")
        m4.metric("Z-Score", f"{z_score:.2f}")

        # Plotly Corridor
        hist_data = s_full[(s_full['Product']==sku) & (s_full['Location']==loc)]['Quantity'].tolist()
        future_data = d_full[(d_full['Product']==sku) & (d_full['Location']==loc)]['Forecast_Quantity'].tolist()
        
        x_axis = [f"H{i}" for i in range(len(hist_data))] + [f"F{i}" for i in range(len(future_data))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=[row['Max_Corridor']]*len(x_axis), mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=x_axis, y=[row['Min_Corridor']]*len(x_axis), fill='tonexty', fillcolor='rgba(0, 123, 255, 0.1)', name='Ideal Corridor', line_color='rgba(0,0,0,0)'))
        fig.add_trace(go.Scatter(x=x_axis[:len(hist_data)], y=hist_data, name='Historical Sales', line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=x_axis[len(hist_data)-1:], y=[hist_data[-1]] + future_data, name='Future Forecast', line=dict(color='blue', dash='dot')))
        
        fig.update_layout(title=f"Inventory Corridor for {sku} at {loc}", hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Network Map: {sku}")
        sku_lt = lt_full[lt_full['Product'] == sku]
        
        net = Network(height="500px", width="100%", directed=True, bgcolor="#ffffff")
        
        # Add Nodes
        all_nodes = set(sku_lt['From_Loc']).union(set(sku_lt['To_Loc']))
        for n in all_nodes:
            color = "#ff4b4b" if n in sku_lt['To_Loc'].values else "#31333F"
            net.add_node(n, label=n, color=color, size=20)
            
        # Add Edges
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Loc'], r['To_Loc'], title=f"LT: {r['Lead_Time_Days']} days", label=f"{r['Lead_Time_Days']}d")
            
        net.save_graph("net.html")
        components.html(open("net.html", 'r').read(), height=550)
        st.caption("🔴 Red = Destination (To_Loc) | ⚫ Black = Source (From_Loc)")

    with tab3:
        st.subheader("Inventory Summary: All Combinations")
        # Modern formatted dataframe
        st.dataframe(
            data[['Product', 'Location', 'Avg_Sales', 'Std_Sales', 'LT_Avg', 'Safety_Stock', 'Min_Corridor', 'Max_Corridor']],
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("🚀 Please upload Sales, Demand, and Lead Time CSVs in the sidebar to begin.")
