import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🛒 SmartCart Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all processed data files"""
    try:
        # Load main datasets
        transactions = pd.read_csv('cleaned_transactions.csv')
        transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'])
        
        rfm_data = pd.read_csv('rfm_customer_segments.csv')
        
        association_rules = pd.read_csv('association_rules.csv')
        
        key_metrics = pd.read_csv('key_metrics.csv')
        
        monthly_trends = pd.read_csv('monthly_trends.csv')
        
        country_performance = pd.read_csv('country_performance.csv')
        
        product_lookup = pd.read_csv('product_lookup.csv')
        
        top_products = pd.read_csv('top_products.csv')
        
        return {
            'transactions': transactions,
            'rfm_data': rfm_data,
            'association_rules': association_rules,
            'key_metrics': key_metrics,
            'monthly_trends': monthly_trends,
            'country_performance': country_performance,
            'product_lookup': product_lookup,
            'top_products': top_products
        }
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.error("Please run the Jupyter notebook analysis first to generate the required data files.")
        return None

def create_kpi_metrics(data):
    """Create KPI metrics display"""
    metrics = data['key_metrics']
    
    # Extract key values
    total_revenue = float(metrics[metrics['metric'] == 'Total Revenue']['value'].iloc[0].replace('$', '').replace(',', ''))
    total_orders = int(metrics[metrics['metric'] == 'Total Orders']['value'].iloc[0].replace(',', ''))
    total_customers = int(metrics[metrics['metric'] == 'Total Customers']['value'].iloc[0].replace(',', ''))
    avg_order_value = float(metrics[metrics['metric'] == 'Average Order Value']['value'].iloc[0].replace('$', ''))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-label">Total Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-label">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">${avg_order_value:.2f}</div>
            <div class="metric-label">Avg Order Value</div>
        </div>
        """, unsafe_allow_html=True)

def create_monthly_trend_chart(data):
    """Create monthly revenue trend chart"""
    monthly_data = data['monthly_trends']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['YearMonth'],
        y=monthly_data['TotalAmount'],
        mode='lines+markers',
        name='Monthly Revenue',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Monthly Revenue Trend",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_top_products_chart(data):
    """Create top products chart"""
    top_products = data['top_products'].head(10)
    
    fig = px.bar(
        top_products,
        x='revenue',
        y='description',
        orientation='h',
        title="Top 10 Products by Revenue",
        labels={'revenue': 'Revenue ($)', 'description': 'Product'},
        color='revenue',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    return fig

def create_customer_segment_chart(data):
    """Create customer segmentation chart"""
    rfm_data = data['rfm_data']
    
    # Create segment summary
    segment_summary = rfm_data.groupby('Cluster_Name').agg({
        'Monetary': ['sum', 'count']
    }).round(2)
    
    segment_summary.columns = ['Total_Revenue', 'Customer_Count']
    segment_summary = segment_summary.reset_index()
    
    fig = px.pie(
        segment_summary,
        values='Customer_Count',
        names='Cluster_Name',
        title="Customer Distribution by Segment",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_country_performance_chart(data):
    """Create country performance chart"""
    country_data = data['country_performance'].head(10)
    
    fig = px.bar(
        country_data,
        x='Country',
        y='TotalAmount',
        title="Top 10 Countries by Revenue",
        labels={'TotalAmount': 'Revenue ($)', 'Country': 'Country'},
        color='TotalAmount',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def product_recommendation_engine(data):
    """Product recommendation engine based on association rules"""
    st.subheader("🛒 Smart Product Recommendation Engine")
    st.markdown("Select products that a customer has purchased to get recommendations for additional items:")
    
    # Get product list
    product_lookup = data['product_lookup']
    association_rules = data['association_rules']
    
    if len(association_rules) == 0:
        st.warning("No association rules available. Please run the analysis with more data.")
        return
    
    # Product selection
    available_products = product_lookup['Description'].unique()[:50]  # Limit for demo
    selected_products = st.multiselect(
        "Select products in customer's basket:",
        available_products,
        max_selections=5
    )
    
    if selected_products:
        st.markdown("### 🎯 Recommended Products:")
        
        # Find recommendations based on association rules
        recommendations = []
        
        for product in selected_products:
            # Find rules where this product is in antecedents
            matching_rules = association_rules[
                association_rules['antecedents_str'].str.contains(product, case=False, na=False)
            ].sort_values('lift', ascending=False).head(3)
            
            for _, rule in matching_rules.iterrows():
                recommendations.append({
                    'recommended_product': rule['consequents_str'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'based_on': rule['antecedents_str']
                })
        
        if recommendations:
            # Display recommendations
            rec_df = pd.DataFrame(recommendations).drop_duplicates('recommended_product')
            
            for i, rec in rec_df.head(5).iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{rec['recommended_product']}**")
                    st.caption(f"Based on: {rec['based_on']}")
                with col2:
                    st.metric("Confidence", f"{rec['confidence']:.1%}")
                with col3:
                    st.metric("Lift", f"{rec['lift']:.2f}")
                st.markdown("---")
        else:
            st.info("No specific recommendations found for the selected products. Try different products or check association rules.")

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🛒 SmartCart Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Predicting Customer Purchase Patterns & Optimizing Store Revenue**")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar
    st.sidebar.markdown("## 📊 Dashboard Controls")
    
    # Dashboard sections
    dashboard_sections = [
        "📈 Business Overview",
        "🛒 Product Recommendations", 
        "👥 Customer Segments",
        "🌍 Geographic Analysis",
        "📋 Detailed Analytics"
    ]
    
    selected_section = st.sidebar.selectbox("Select Dashboard Section:", dashboard_sections)
    
    # Sidebar info
    st.sidebar.markdown("""
    <div class="sidebar-info">
    <h4>ℹ️ About This Dashboard</h4>
    <p>This interactive dashboard provides insights from retail transaction analysis including:</p>
    <ul>
    <li>📊 Key performance metrics</li>
    <li>🛒 Product associations</li>
    <li>👥 Customer segmentation</li>
    <li>🌍 Geographic performance</li>
    <li>💡 Business recommendations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content based on selection
    if selected_section == "📈 Business Overview":
        st.header("📈 Business Performance Overview")
        
        # KPI Metrics
        create_kpi_metrics(data)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_chart = create_monthly_trend_chart(data)
            st.plotly_chart(monthly_chart, use_container_width=True)
        
        with col2:
            top_products_chart = create_top_products_chart(data)
            st.plotly_chart(top_products_chart, use_container_width=True)
        
        # Additional insights
        st.subheader("📊 Key Insights")
        insights = data['key_metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_day = insights[insights['metric'] == 'Best Performing Day']['value'].iloc[0]
            st.info(f"🗓️ **Best Sales Day:** {best_day}")
        
        with col2:
            peak_season = insights[insights['metric'] == 'Peak Season']['value'].iloc[0]
            st.info(f"🌟 **Peak Season:** {peak_season}")
        
        with col3:
            top_country = insights[insights['metric'] == 'Top Country']['value'].iloc[0]
            st.info(f"🌍 **Top Market:** {top_country}")
    
    elif selected_section == "🛒 Product Recommendations":
        product_recommendation_engine(data)
        
        # Association rules table
        st.subheader("🔗 Top Product Associations")
        if len(data['association_rules']) > 0:
            rules_display = data['association_rules'].head(10).copy()
            rules_display['confidence'] = rules_display['confidence'].apply(lambda x: f"{x:.1%}")
            rules_display['lift'] = rules_display['lift'].apply(lambda x: f"{x:.2f}")
            rules_display['support'] = rules_display['support'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(
                rules_display[['antecedents_str', 'consequents_str', 'confidence', 'lift', 'support']],
                column_config={
                    'antecedents_str': 'Product A',
                    'consequents_str': 'Product B (Recommended)',
                    'confidence': 'Confidence',
                    'lift': 'Lift',
                    'support': 'Support'
                },
                use_container_width=True
            )
        else:
            st.warning("No association rules available.")
    
    elif selected_section == "👥 Customer Segments":
        st.header("👥 Customer Segmentation Analysis")
        
        # Customer segment chart
        segment_chart = create_customer_segment_chart(data)
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.plotly_chart(segment_chart, use_container_width=True)
        
        with col2:
            st.subheader("Segment Details")
            rfm_summary = data['rfm_data'].groupby('Cluster_Name').agg({
                'Monetary': ['sum', 'mean', 'count'],
                'Frequency': 'mean',
                'Recency': 'mean'
            }).round(2)
            
            rfm_summary.columns = ['Total Revenue', 'Avg Revenue', 'Customers', 'Avg Frequency', 'Avg Recency']
            st.dataframe(rfm_summary, use_container_width=True)
        
        # RFM Distribution
        st.subheader("📊 RFM Score Distribution")
        rfm_data = data['rfm_data']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_r = px.histogram(rfm_data, x='R_Score', title='Recency Score Distribution')
            st.plotly_chart(fig_r, use_container_width=True)
        
        with col2:
            fig_f = px.histogram(rfm_data, x='F_Score', title='Frequency Score Distribution')
            st.plotly_chart(fig_f, use_container_width=True)
        
        with col3:
            fig_m = px.histogram(rfm_data, x='M_Score', title='Monetary Score Distribution')
            st.plotly_chart(fig_m, use_container_width=True)
    
    elif selected_section == "🌍 Geographic Analysis":
        st.header("🌍 Geographic Performance Analysis")
        
        # Country performance chart
        country_chart = create_country_performance_chart(data)
        st.plotly_chart(country_chart, use_container_width=True)
        
        # Detailed country table
        st.subheader("📊 Detailed Country Performance")
        country_data = data['country_performance'].copy()
        
        # Format columns for display
        country_data['TotalAmount'] = country_data['TotalAmount'].apply(lambda x: f"${x:,.2f}")
        country_data['Avg_Order_Value'] = country_data['Avg_Order_Value'].apply(lambda x: f"${x:.2f}")
        country_data['Customer_Value'] = country_data['Customer_Value'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            country_data[['Country', 'TotalAmount', 'CustomerID', 'InvoiceNo', 'Avg_Order_Value', 'Customer_Value']],
            column_config={
                'Country': 'Country',
                'TotalAmount': 'Total Revenue',
                'CustomerID': 'Unique Customers',
                'InvoiceNo': 'Total Orders',
                'Avg_Order_Value': 'Avg Order Value',
                'Customer_Value': 'Avg Customer Value'
            },
            use_container_width=True
        )
    
    elif selected_section == "📋 Detailed Analytics":
        st.header("📋 Detailed Analytics & Raw Data")
        
        # Data selection
        data_option = st.selectbox(
            "Select dataset to explore:",
            ["Transaction Data", "Customer Segments", "Product Performance", "Association Rules"]
        )
        
        if data_option == "Transaction Data":
            st.subheader("🛍️ Transaction Data")
            st.dataframe(data['transactions'].head(1000), use_container_width=True)
            st.caption(f"Showing first 1000 rows of {len(data['transactions']):,} total transactions")
        
        elif data_option == "Customer Segments":
            st.subheader("👥 Customer Segments Data") 
            st.dataframe(data['rfm_data'], use_container_width=True)
        
        elif data_option == "Product Performance":
            st.subheader("📦 Product Performance Data")
            st.dataframe(data['top_products'], use_container_width=True)
        
        elif data_option == "Association Rules":
            st.subheader("🔗 Association Rules Data")
            if len(data['association_rules']) > 0:
                st.dataframe(data['association_rules'], use_container_width=True)
            else:
                st.warning("No association rules data available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🛒 SmartCart Analytics Dashboard | Built with Streamlit & Python</p>
    <p>📊 Data-driven insights for retail optimization and customer intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()