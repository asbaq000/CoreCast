import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_processing import load_and_preprocess_data, create_features
from modeling import train_evaluate_arima, train_evaluate_random_forest
from utils import calculate_evaluation_metrics

st.set_page_config(page_title="CoreCast", layout="wide", initial_sidebar_state="expanded")

if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'fitted_model' not in st.session_state:
    st.session_state.fitted_model = None
if 'model_option' not in st.session_state:
    st.session_state.model_option = "ARIMA"
if 'full_featured_data' not in st.session_state:
    st.session_state.full_featured_data = pd.DataFrame()

st.title("Automated Supply Chain Demand Forecasting")
st.markdown("A professional tool to help local businesses predict demand and optimize inventory.")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("1. Upload Your Sales CSV", type=["csv"])

if uploaded_file:
    data = load_and_preprocess_data(uploaded_file)
    
    if not data.empty:
        st.session_state.full_featured_data = create_features(data)
        
        st.header("Processed Sales Data Preview")
        st.dataframe(data.head())

        model_option = st.sidebar.selectbox(
            "2. Choose a Forecasting Model",
            ("ARIMA", "Random Forest"),
            key="model_option"
        )
        
        train_df, test_df = st.session_state.full_featured_data.iloc[:int(len(st.session_state.full_featured_data) * 0.8)], st.session_state.full_featured_data.iloc[int(len(st.session_state.full_featured_data) * 0.8):]

        if st.sidebar.button("3. Generate Forecast on Test Data", key="generate", use_container_width=True):
            with st.spinner(f"Training {model_option} model and evaluating on test data..."):
                forecast_series = None
                if model_option == "ARIMA":
                    fitted_model, forecast_series = train_evaluate_arima(train_df['Sales'], test_df['Sales'])
                    st.session_state.fitted_model = fitted_model

                elif model_option == "Random Forest":
                    fitted_model, forecast_series = train_evaluate_random_forest(train_df, test_df)
                    st.session_state.fitted_model = fitted_model
                
            st.success("Evaluation forecast generated successfully!")
            st.session_state.forecast_df = pd.DataFrame({'Actual': test_df['Sales'], 'Forecast': forecast_series})

        if st.session_state.forecast_df is not None:
            results_df = st.session_state.forecast_df
            st.header("Forecast vs. Actual Sales (Evaluation on Test Set)")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Sales'], mode='lines', name='Historical Sales', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual Sales (Test Set)', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Forecast'], mode='lines', name='Forecasted Sales', line=dict(color='red', dash='dash')))
            fig.update_layout(title="Sales Forecast Analysis", xaxis_title="Date", yaxis_title="Sales Volume", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)

            st.header("Model Performance Metrics")
            metrics = calculate_evaluation_metrics(results_df['Actual'], results_df['Forecast'])
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

        st.sidebar.header("4. Generate Future Forecast")
        future_periods = st.sidebar.number_input("Enter number of days to forecast:", min_value=7, max_value=365, value=30)
        
        if st.sidebar.button("Forecast Future", key="future_forecast", use_container_width=True):
            if st.session_state.fitted_model is None:
                st.sidebar.error(f"A {model_option} model has not been trained yet. Please run Step 3.")
            else:
                with st.spinner("Generating future forecast..."):
                    future_forecast = None
                    if model_option == "ARIMA":
                        future_forecast = st.session_state.fitted_model.forecast(steps=future_periods)
                        future_dates = pd.date_range(start=st.session_state.full_featured_data.index[-1] + pd.Timedelta(days=1), periods=future_periods)
                        future_df = pd.DataFrame({'Forecast': future_forecast}, index=future_dates)
                    
                    elif model_option == "Random Forest":
                        features_list = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag7', 'rolling_mean_7']
                        history = st.session_state.full_featured_data.copy()
                        future_predictions = []
                        
                        for i in range(future_periods):
                            last_date = history.index[-1]
                            next_date = last_date + pd.Timedelta(days=1)
                            
                            next_step_features = pd.DataFrame(index=[next_date])
                            next_step_features['dayofweek'] = next_step_features.index.dayofweek
                            next_step_features['quarter'] = next_step_features.index.quarter
                            next_step_features['month'] = next_step_features.index.month
                            next_step_features['year'] = next_step_features.index.year
                            next_step_features['dayofyear'] = next_step_features.index.dayofyear
                            next_step_features['lag1'] = history['Sales'].iloc[-1]
                            next_step_features['lag7'] = history['Sales'].iloc[-7]
                            next_step_features['rolling_mean_7'] = history['Sales'].rolling(window=7).mean().iloc[-1]
                            
                            prediction = st.session_state.fitted_model.predict(next_step_features[features_list])[0]
                            future_predictions.append(prediction)
                            
                            new_row = pd.DataFrame({'Sales': [prediction]}, index=[next_date])
                            history = pd.concat([history, new_row])
                        
                        future_df = pd.DataFrame({'Forecast': future_predictions}, index=history.index[-future_periods:])

                if 'future_df' in locals() and not future_df.empty:
                    st.header(f"Future Demand Forecast for the Next {future_periods} Days")
                    
                    fig_future = go.Figure()
                    fig_future.add_trace(go.Scatter(x=data.index, y=data['Sales'], mode='lines', name='Historical Sales'))
                    fig_future.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='Future Forecast', line=dict(color='green')))
                    fig_future.update_layout(title="Future Demand", xaxis_title="Date", yaxis_title="Predicted Sales")
                    st.plotly_chart(fig_future, use_container_width=True)
                    st.dataframe(future_df)
else:
    st.info("Upload a CSV file to begin the forecasting process.")
