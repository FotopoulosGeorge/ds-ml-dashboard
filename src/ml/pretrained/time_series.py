# src/ml/pretrained/time_series.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet import with fallback
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class TimeSeriesForecaster:
    """
    Time series forecasting using Prophet and other models
    """
    
    def __init__(self):
        self.model = None
        self.forecast = None
        self.data = None
    
    def render_time_series_tab(self, df):
        """
        Main time series interface
        """
        st.header("📈 **Time Series Forecasting**")
        
        if not PROPHET_AVAILABLE:
            st.error("❌ Prophet unavailable in Demo Mode")
            st.info("💡 Prophet is Facebook's time series forecasting tool - excellent for business data!")
            return
        
        st.markdown("*Forecast future values using advanced time series models*")
        
        # Data validation
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not date_cols:
            st.warning("⚠️ No datetime columns found. Prophet needs a date column.")
            st.info("💡 **Tip:** Convert your date column using Feature Engineering → Date Features")
            return
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for forecasting.")
            return
        
        # Configuration
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            date_col = st.selectbox(
                "**Date Column:**",
                date_cols,
                key="ts_date_col"
            )
            
            target_col = st.selectbox(
                "**Value to Forecast:**",
                numeric_cols,
                key="ts_target_col"
            )
        
        with config_col2:
            forecast_periods = st.slider(
                "**Forecast Periods:**",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of future periods to predict",
                key="ts_periods"
            )
            
            freq = st.selectbox(
                "**Frequency:**",
                ["D", "W", "M", "Q", "Y"],
                index=0,
                format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}[x],
                key="ts_freq"
            )
        
        # Data preprocessing
        if st.button("📈 **Generate Forecast**", type="primary", key="generate_forecast"):
            try:
                with st.spinner('Building forecast model...'):
                    forecast_result = self._create_prophet_forecast(
                        df, date_col, target_col, forecast_periods, freq
                    )
                
                if forecast_result:
                    self._display_forecast_results(forecast_result)
                    
            except Exception as e:
                st.error(f"❌ Forecasting failed: {str(e)}")
                st.info("💡 **Common issues:** Irregular dates, too few data points, or missing values")
        
        # Advanced options
        with st.expander("⚙️ **Advanced Prophet Settings**"):
            st.markdown("**Seasonality & Trends:**")
            
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                yearly_seasonality = st.checkbox("Yearly Seasonality", value=True, key="yearly_season")
                weekly_seasonality = st.checkbox("Weekly Seasonality", value=True, key="weekly_season")
                
            with adv_col2:
                daily_seasonality = st.checkbox("Daily Seasonality", value=False, key="daily_season")
                uncertainty_samples = st.slider("Uncertainty Samples:", 0, 1000, 100, key="uncertainty")
            
            st.markdown("**Growth & Changepoints:**")
            growth = st.selectbox("Growth Type:", ["linear", "logistic"], key="growth_type")
            
            if growth == "logistic":
                st.info("💡 For logistic growth, set capacity limits in your data")
    
    def _create_prophet_forecast(self, df, date_col, target_col, periods, freq):
        """
        Create Prophet forecast
        """
        # Prepare data for Prophet
        ts_data = df[[date_col, target_col]].copy()
        ts_data = ts_data.dropna()
        ts_data = ts_data.rename(columns={date_col: 'ds', target_col: 'y'})
        ts_data = ts_data.sort_values('ds')
        
        # Validate data
        if len(ts_data) < 10:
            st.error("❌ Need at least 10 data points for forecasting")
            return None
        
        # Initialize Prophet model
        model_params = {
            'yearly_seasonality': st.session_state.get('yearly_season', True),
            'weekly_seasonality': st.session_state.get('weekly_season', True),
            'daily_seasonality': st.session_state.get('daily_season', False),
            'uncertainty_samples': st.session_state.get('uncertainty', 100)
        }
        
        model = Prophet(**model_params)
        
        # Fit model
        model.fit(ts_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Store results
        self.model = model
        self.forecast = forecast
        self.data = ts_data
        
        return {
            'model': model,
            'forecast': forecast,
            'historical_data': ts_data,
            'future_dates': future,
            'target_col': target_col,
            'date_col': date_col,
            'periods': periods
        }
    
    def _display_forecast_results(self, result):
        """
        Display comprehensive forecast results
        """
        st.success("✅ Forecast generated successfully!")
        
        model = result['model']
        forecast = result['forecast']
        historical_data = result['historical_data']
        target_col = result['target_col']
        periods = result['periods']
        
        # Key metrics
        forecast_start = len(historical_data)
        future_forecast = forecast[forecast_start:]
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Historical Points", len(historical_data))
        with metric_col2:
            st.metric("Forecast Points", len(future_forecast))
        with metric_col3:
            avg_future = future_forecast['yhat'].mean()
            st.metric("Avg Future Value", f"{avg_future:.2f}")
        with metric_col4:
            trend_direction = "📈" if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else "📉"
            st.metric("Trend", trend_direction)
        
        # Main forecast plot
        st.subheader("📈 Forecast Visualization")
        
        try:
            # Create plotly figure
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data['ds'],
                y=historical_data['y'],
                mode='markers+lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            # Add vertical line to separate historical from forecast
            last_historical_date = historical_data['ds'].iloc[-1]
            fig.add_vline(
                x=last_historical_date,
                line_dash="dot",
                line_color="green",
                annotation_text="Forecast Start"
            )
            
            fig.update_layout(
                title=f'{target_col} Forecast',
                xaxis_title='Date',
                yaxis_title=target_col,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Visualization error: {str(e)}")
        
        # Components plot
        st.subheader("📊 Forecast Components")
        
        try:
            components_fig = plot_components_plotly(model, forecast)
            st.plotly_chart(components_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Components plot error: {str(e)}")
        
        # Forecast table
        st.subheader("📋 Forecast Data")
        
        # Show future forecast data
        display_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        display_forecast.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_forecast, use_container_width=True)
        
        # Download forecast
        csv_data = display_forecast.to_csv(index=False)
        st.download_button(
            label="📥 **Download Forecast**",
            data=csv_data,
            file_name=f"forecast_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Model performance on historical data
        if len(historical_data) > 10:
            st.subheader("🎯 Model Performance")
            self._display_model_performance(model, historical_data)
    
    def _display_model_performance(self, model, historical_data):
        """
        Show model performance metrics on historical data
        """
        try:
            # Cross validation on historical data
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Simple performance metrics
            forecast_hist = model.predict(historical_data[['ds']])
            
            actual = historical_data['y']
            predicted = forecast_hist['yhat']
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("MAE", f"{mae:.2f}")
            with perf_col2:
                st.metric("MAPE", f"{mape:.2f}%")
            with perf_col3:
                st.metric("RMSE", f"{rmse:.2f}")
            
            # Residuals plot
            residuals = actual - predicted
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data['ds'],
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='purple')
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title='Model Residuals (Actual - Predicted)',
                xaxis_title='Date',
                yaxis_title='Residuals'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.info(f"Performance analysis unavailable: {str(e)}")
    
    def get_forecast_summary(self):
        """
        Get summary of the last forecast for saving/export
        """
        if self.forecast is None:
            return None
        
        return {
            'model_type': 'Prophet Time Series',
            'forecast_points': len(self.forecast),
            'created_at': datetime.now().isoformat(),
            'forecast_data': self.forecast.to_dict(),
            'model_params': self.model.get_params() if self.model else None
        }