"""
=============================================================================
FAST FOOD RESTAURANT QUEUE SIMULATION - STREAMLIT APP
Discrete Event Simulation using SimPy
=============================================================================
Author: Putra Aliansyah
NIM: 301230041
Mata Kuliah: Pembelajaran Mesin - Pemodelan dan Simulasi
Dosen: Mohammad Bayu Anggara, S.Kom., M.Kom.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import simpy
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Set page config
st.set_page_config(
    page_title="Fast Food Queue Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

class QueueSimulator:
    """Class untuk menjalankan simulasi antrean"""
    
    def __init__(self):
        self.waiting_times = []
        self.queue_lengths = []
        self.system_times = []
        self.arrivals = []
        self.queue_over_time = []
        self.time_stamps = []
        
    def customer_process(self, env, name, server, service_mean, service_std):
        """Proses yang dilalui customer"""
        arrival_time = env.now
        self.arrivals.append(arrival_time)
        
        with server.request() as request:
            # Catat panjang antrean
            queue_len = len(server.queue)
            self.queue_lengths.append(queue_len)
            self.queue_over_time.append(queue_len)
            self.time_stamps.append(env.now)
            
            yield request
            
            # Waktu tunggu
            waiting_time = env.now - arrival_time
            self.waiting_times.append(waiting_time)
            
            # Service time
            service_time = max(0.5, np.random.normal(service_mean, service_std))
            yield env.timeout(service_time)
            
            # Total waktu di sistem
            system_time = env.now - arrival_time
            self.system_times.append(system_time)
            
    def customer_generator(self, env, server, arrival_rate, service_mean, service_std, num_customers):
        """Generator kedatangan customer"""
        for i in range(num_customers):
            inter_arrival = np.random.exponential(1 / arrival_rate)
            yield env.timeout(inter_arrival)
            env.process(self.customer_process(env, f'Customer_{i+1}', server, service_mean, service_std))
    
    def run_simulation(self, num_servers, sim_duration, arrival_rate, service_mean, service_std, num_customers):
        """Menjalankan simulasi"""
        # Reset metrics
        self.waiting_times = []
        self.queue_lengths = []
        self.system_times = []
        self.arrivals = []
        self.queue_over_time = []
        self.time_stamps = []
        
        # Setup environment
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=num_servers)
        
        # Start generator
        env.process(self.customer_generator(env, server, arrival_rate, service_mean, service_std, num_customers))
        
        # Run simulation
        env.run(until=sim_duration)
        
        # Calculate metrics
        results = {
            'num_servers': num_servers,
            'avg_waiting_time': np.mean(self.waiting_times) if self.waiting_times else 0,
            'max_waiting_time': np.max(self.waiting_times) if self.waiting_times else 0,
            'std_waiting_time': np.std(self.waiting_times) if self.waiting_times else 0,
            'avg_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': np.max(self.queue_lengths) if self.queue_lengths else 0,
            'avg_system_time': np.mean(self.system_times) if self.system_times else 0,
            'total_customers': len(self.waiting_times),
            'customers_with_wait': sum(1 for wt in self.waiting_times if wt > 0),
            'percent_with_wait': (sum(1 for wt in self.waiting_times if wt > 0) / len(self.waiting_times) * 100) if self.waiting_times else 0
        }
        
        return results

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header"> Fast Food Restaurant Queue Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discrete Event Simulation using SimPy</p>', unsafe_allow_html=True)
    
    # Sidebar - Input Parameters
    st.sidebar.header("⚙️ Simulation Parameters")
    
    st.sidebar.markdown("### 📊 System Configuration")
    num_servers = st.sidebar.slider(
        "Number of Servers (Cashiers)",
        min_value=1,
        max_value=10,
        value=3,
        help="Jumlah kasir yang melayani pelanggan"
    )
    
    st.sidebar.markdown("### 👥 Customer Arrival")
    arrival_rate = st.sidebar.slider(
        "Arrival Rate (customers/minute)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Rata-rata jumlah pelanggan yang datang per menit"
    )
    
    st.sidebar.markdown("### ⏱️ Service Time")
    service_mean = st.sidebar.slider(
        "Mean Service Time (minutes)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Rata-rata waktu yang dibutuhkan untuk melayani 1 pelanggan"
    )
    
    service_std = st.sidebar.slider(
        "Std Dev Service Time (minutes)",
        min_value=0.1,
        max_value=3.0,
        value=0.8,
        step=0.1,
        help="Variasi waktu layanan"
    )
    
    st.sidebar.markdown("### 🕐 Simulation Settings")
    sim_duration = st.sidebar.number_input(
        "Simulation Duration (minutes)",
        min_value=60,
        max_value=1440,
        value=480,
        step=60,
        help="Durasi simulasi dalam menit (480 = 8 jam)"
    )
    
    num_customers = st.sidebar.number_input(
        "Expected Number of Customers",
        min_value=50,
        max_value=1000,
        value=250,
        step=50,
        help="Estimasi jumlah pelanggan yang akan datang"
    )
    
    st.sidebar.markdown("### 💰 Cost Parameters (IDR)")
    hourly_wage = st.sidebar.number_input(
        "Hourly Wage per Server",
        min_value=10000,
        max_value=100000,
        value=25000,
        step=5000,
        help="Gaji kasir per jam"
    )
    
    customer_loss = st.sidebar.number_input(
        "Customer Loss per Minute Waiting",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Estimasi kerugian per menit pelanggan menunggu"
    )
    
    # Run Simulation Button
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Info Box - System Overview
    with st.expander("ℹ️ About This Simulator", expanded=False):
        st.markdown("""
        **Fast Food Queue Simulator** adalah aplikasi simulasi discrete event yang mensimulasikan 
        sistem antrean pada restoran cepat saji menggunakan framework **SimPy**.
        
        **Fitur Utama:**
        - 🎯 Simulasi real-time dengan berbagai parameter
        - 📊 Visualisasi interaktif hasil simulasi
        - 💰 Analisis cost-benefit untuk optimasi
        - 📈 Metrik performa lengkap
        - 🔄 Perbandingan multiple skenario
        
        **Cara Penggunaan:**
        1. Atur parameter di sidebar kiri
        2. Klik tombol "Run Simulation"
        3. Lihat hasil dan analisis
        4. Eksperimen dengan berbagai skenario untuk menemukan konfigurasi optimal
        """)
    
    # Main Content
    if run_simulation:
        with st.spinner('🔄 Running simulation... Please wait...'):
            # Initialize simulator
            simulator = QueueSimulator()
            
            # Run single simulation
            results = simulator.run_simulation(
                num_servers=num_servers,
                sim_duration=sim_duration,
                arrival_rate=arrival_rate,
                service_mean=service_mean,
                service_std=service_std,
                num_customers=num_customers
            )
            
            # Display Success Message
            st.success('✅ Simulation completed successfully!')
            
            # =================================================================
            # KEY METRICS
            # =================================================================
            st.markdown("## 📊 Key Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="👥 Total Customers Served",
                    value=f"{results['total_customers']}",
                    delta=f"{results['customers_with_wait']} waited"
                )
            
            with col2:
                st.metric(
                    label="⏱️ Avg Waiting Time",
                    value=f"{results['avg_waiting_time']:.2f} min",
                    delta=f"Max: {results['max_waiting_time']:.2f} min",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    label="📏 Avg Queue Length",
                    value=f"{results['avg_queue_length']:.2f}",
                    delta=f"Max: {int(results['max_queue_length'])}"
                )
            
            with col4:
                st.metric(
                    label="🕐 Avg System Time",
                    value=f"{results['avg_system_time']:.2f} min",
                    delta=f"{results['percent_with_wait']:.1f}% waited"
                )
            
            # =================================================================
            # VISUALIZATIONS
            # =================================================================
            st.markdown("## 📈 Visualization Dashboard")
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Time Series", "💰 Cost Analysis", "🔄 Scenario Comparison"])
            
            with tab1:
                st.markdown("### Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Waiting Time Distribution
                    fig_wait = go.Figure()
                    fig_wait.add_trace(go.Histogram(
                        x=simulator.waiting_times,
                        nbinsx=30,
                        name='Waiting Time',
                        marker_color='#FF6B6B',
                        opacity=0.7
                    ))
                    fig_wait.add_vline(
                        x=results['avg_waiting_time'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {results['avg_waiting_time']:.2f} min"
                    )
                    fig_wait.update_layout(
                        title="Waiting Time Distribution",
                        xaxis_title="Waiting Time (minutes)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_wait, use_container_width=True)
                
                with col2:
                    # System Time Distribution
                    fig_system = go.Figure()
                    fig_system.add_trace(go.Histogram(
                        x=simulator.system_times,
                        nbinsx=30,
                        name='System Time',
                        marker_color='#4ECDC4',
                        opacity=0.7
                    ))
                    fig_system.add_vline(
                        x=results['avg_system_time'],
                        line_dash="dash",
                        line_color="blue",
                        annotation_text=f"Mean: {results['avg_system_time']:.2f} min"
                    )
                    fig_system.update_layout(
                        title="System Time Distribution",
                        xaxis_title="System Time (minutes)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_system, use_container_width=True)
                
                # Queue Length Distribution
                fig_queue = go.Figure()
                fig_queue.add_trace(go.Histogram(
                    x=simulator.queue_lengths,
                    nbinsx=max(simulator.queue_lengths) + 1 if simulator.queue_lengths else 10,
                    name='Queue Length',
                    marker_color='#95E1D3',
                    opacity=0.7
                ))
                fig_queue.update_layout(
                    title="Queue Length Distribution",
                    xaxis_title="Queue Length (number of customers)",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_queue, use_container_width=True)
            
            with tab2:
                st.markdown("### Time Series Analysis")
                
                # Queue Length Over Time
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=simulator.time_stamps,
                    y=simulator.queue_over_time,
                    mode='lines',
                    name='Queue Length',
                    line=dict(color='#FF6B6B', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.2)'
                ))
                fig_time.add_hline(
                    y=results['avg_queue_length'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Average: {results['avg_queue_length']:.2f}"
                )
                fig_time.update_layout(
                    title="Queue Length Evolution Over Time",
                    xaxis_title="Time (minutes)",
                    yaxis_title="Queue Length",
                    showlegend=False,
                    height=500
                )
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Customer Arrivals
                fig_arrivals = go.Figure()
                fig_arrivals.add_trace(go.Scatter(
                    x=list(range(len(simulator.arrivals))),
                    y=simulator.arrivals,
                    mode='markers',
                    name='Arrivals',
                    marker=dict(color='#4ECDC4', size=6),
                ))
                fig_arrivals.update_layout(
                    title="Customer Arrival Pattern",
                    xaxis_title="Customer Number",
                    yaxis_title="Arrival Time (minutes)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_arrivals, use_container_width=True)
            
            with tab3:
                st.markdown("### Cost-Benefit Analysis")
                
                # Calculate costs
                operational_hours = sim_duration / 60
                server_cost = num_servers * hourly_wage * operational_hours
                waiting_cost = results['avg_waiting_time'] * results['total_customers'] * customer_loss
                total_cost = server_cost + waiting_cost
                
                # Display cost metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**💵 Server Cost**")
                    st.markdown(f"## Rp {server_cost:,.0f}")
                    st.markdown(f"_{num_servers} servers × {operational_hours:.1f} hours_")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**⏰ Waiting Cost**")
                    st.markdown(f"## Rp {waiting_cost:,.0f}")
                    st.markdown(f"_Customer waiting time penalty_")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**💰 Total Cost**")
                    st.markdown(f"## Rp {total_cost:,.0f}")
                    st.markdown(f"_Total operational cost_")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Cost Breakdown Pie Chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Server Cost', 'Waiting Cost'],
                    values=[server_cost, waiting_cost],
                    marker_colors=['#4ECDC4', '#FF6B6B'],
                    hole=.4
                )])
                fig_pie.update_layout(
                    title="Cost Breakdown",
                    height=400,
                    annotations=[dict(text=f'Rp {total_cost/1000:.0f}K', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab4:
                st.markdown("### Scenario Comparison")
                st.markdown("Compare different numbers of servers to find the optimal configuration.")
                
                # Run multiple scenarios
                scenarios = range(max(1, num_servers - 2), min(11, num_servers + 3))
                scenario_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, servers in enumerate(scenarios):
                    status_text.text(f"Running scenario: {servers} servers...")
                    sim = QueueSimulator()
                    res = sim.run_simulation(servers, sim_duration, arrival_rate, service_mean, service_std, num_customers)
                    
                    # Calculate costs
                    sc = servers * hourly_wage * (sim_duration / 60)
                    wc = res['avg_waiting_time'] * res['total_customers'] * customer_loss
                    tc = sc + wc
                    
                    res['server_cost'] = sc
                    res['waiting_cost'] = wc
                    res['total_cost'] = tc
                    
                    scenario_results.append(res)
                    progress_bar.progress((idx + 1) / len(scenarios))
                
                status_text.text("✅ All scenarios completed!")
                progress_bar.empty()
                
                # Convert to DataFrame
                df_scenarios = pd.DataFrame(scenario_results)
                
                # Find optimal
                optimal_idx = df_scenarios['total_cost'].idxmin()
                optimal_servers = df_scenarios.loc[optimal_idx, 'num_servers']
                
                st.success(f"🎯 **Optimal Configuration:** {int(optimal_servers)} servers with total cost Rp {df_scenarios.loc[optimal_idx, 'total_cost']:,.0f}")
                
                # Comparison Charts
                fig_comparison = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Waiting Time vs Servers', 'Queue Length vs Servers', 
                                   'Total Cost vs Servers', 'Cost Breakdown'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"type": "bar"}]]
                )
                
                # Plot 1: Waiting Time
                fig_comparison.add_trace(
                    go.Scatter(x=df_scenarios['num_servers'], y=df_scenarios['avg_waiting_time'],
                              mode='lines+markers', name='Avg Waiting Time', line=dict(color='#FF6B6B', width=3)),
                    row=1, col=1
                )
                
                # Plot 2: Queue Length
                fig_comparison.add_trace(
                    go.Scatter(x=df_scenarios['num_servers'], y=df_scenarios['max_queue_length'],
                              mode='lines+markers', name='Max Queue Length', line=dict(color='#4ECDC4', width=3)),
                    row=1, col=2
                )
                
                # Plot 3: Total Cost
                fig_comparison.add_trace(
                    go.Scatter(x=df_scenarios['num_servers'], y=df_scenarios['total_cost'],
                              mode='lines+markers', name='Total Cost', line=dict(color='#95E1D3', width=3)),
                    row=2, col=1
                )
                fig_comparison.add_vline(x=optimal_servers, line_dash="dash", line_color="red", row=2, col=1,
                                        annotation_text="Optimal")
                
                # Plot 4: Cost Breakdown (Stacked Bar)
                fig_comparison.add_trace(
                    go.Bar(x=df_scenarios['num_servers'], y=df_scenarios['server_cost'],
                          name='Server Cost', marker_color='#4ECDC4'),
                    row=2, col=2
                )
                fig_comparison.add_trace(
                    go.Bar(x=df_scenarios['num_servers'], y=df_scenarios['waiting_cost'],
                          name='Waiting Cost', marker_color='#FF6B6B'),
                    row=2, col=2
                )
                
                fig_comparison.update_xaxes(title_text="Number of Servers", row=1, col=1)
                fig_comparison.update_xaxes(title_text="Number of Servers", row=1, col=2)
                fig_comparison.update_xaxes(title_text="Number of Servers", row=2, col=1)
                fig_comparison.update_xaxes(title_text="Number of Servers", row=2, col=2)
                
                fig_comparison.update_yaxes(title_text="Minutes", row=1, col=1)
                fig_comparison.update_yaxes(title_text="Customers", row=1, col=2)
                fig_comparison.update_yaxes(title_text="IDR", row=2, col=1)
                fig_comparison.update_yaxes(title_text="IDR", row=2, col=2)
                
                fig_comparison.update_layout(height=800, showlegend=True, barmode='stack')
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Data Table
                st.markdown("### 📋 Detailed Scenario Results")
                display_df = df_scenarios[['num_servers', 'avg_waiting_time', 'max_queue_length', 
                                          'total_customers', 'total_cost']].copy()
                display_df.columns = ['Servers', 'Avg Wait (min)', 'Max Queue', 'Customers', 'Total Cost (IDR)']
                display_df['Total Cost (IDR)'] = display_df['Total Cost (IDR)'].apply(lambda x: f"Rp {x:,.0f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            

    
    else:
        # Initial state - Show instructions
        st.info("👈 **Atur parameter simulasi di sidebar, lalu klik 'Run Simulation' untuk memulai!**")
        
        st.markdown("""
        ### 📖 Panduan Penggunaan
        
        1. **Atur Parameter:**
           - **Number of Servers**: Jumlah kasir yang melayani
           - **Arrival Rate**: Seberapa cepat pelanggan datang (customers/menit)
           - **Service Time**: Berapa lama melayani 1 pelanggan
           - **Simulation Duration**: Berapa lama simulasi berjalan (dalam menit)
        
        2. **Jalankan Simulasi:**
           - Klik tombol "🚀 Run Simulation"
           - Tunggu proses selesai
        
        3. **Analisis Hasil:**
           - Lihat metrik performa utama
           - Eksplorasi visualisasi di berbagai tab
           - Bandingkan berbagai skenario di tab "Scenario Comparison"
        
        4. **Optimasi:**
           - Cari konfigurasi optimal yang meminimalkan total cost
           - Balance antara biaya kasir vs kerugian waiting time pelanggan
        
        5. **Download:**
           - Download hasil dalam format JSON atau CSV untuk laporan
        """)
        
        # Example scenarios
        st.markdown("### 💡 Contoh Skenario")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🌅 Low Traffic (Pagi)**
            - Arrival Rate: 0.2-0.3
            - Servers: 2-3
            - Service: 2-3 menit
            """)
        
        with col2:
            st.markdown("""
            **☀️ Normal Traffic (Siang)**
            - Arrival Rate: 0.4-0.6
            - Servers: 3-4
            - Service: 2.5-3.5 menit
            """)
        
        with col3:
            st.markdown("""
            **🌆 Peak Hours (Malam)**
            - Arrival Rate: 0.7-1.0
            - Servers: 4-5
            - Service: 3-4 menit
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p><strong>Fast Food Restaurant Queue Simulator</strong></p>
            <p>Developed using SimPy & Streamlit | Tugas Besar Pemodelan dan Simulasi</p>
            <p>© 2025 - [Putra Aliansyah] | </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
