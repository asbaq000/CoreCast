import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from data_processing import load_and_preprocess_data, create_features
from modeling import train_evaluate_arima, train_evaluate_random_forest, train_evaluate_lstm
from utils import calculate_evaluation_metrics

st.set_page_config(page_title="CoreCast", layout="wide")

st.title("Automated Supply Chain Demand Forecasting")
st.markdown("A professional tool to help local businesses predict demand and optimize inventory.")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("1. Upload Your Sales CSV", type=["csv"])

if uploaded_file:
    data = load_and_preprocess_data(uploaded_file)
    if not data.empty:
        original_data = data.copy()
        
        st.header("Raw Sales Data Preview")
        st.dataframe(data.head())

        data_featured = create_features(data)

        model_option = st.sidebar.selectbox(
            "2. Choose a Forecasting Model",
            ("ARIMA", "Random Forest", "LSTM")
        )
        
        train_df, test_df = data_featured.iloc[:int(len(data_featured) * 0.8)], data_featured.iloc[int(len(data_featured) * 0.8):]

        if 'forecast_df' not in st.session_state:
            st.session_state.forecast_df = None
        if 'fitted_model' not in st.session_state:
            st.session_state.fitted_model = None
            
        if st.sidebar.button("3. Generate Forecast on Test Data", key="generate"):
            with st.spinner(f"Training {model_option} model and evaluating on test data..."):
                forecast = None

                if model_option == "ARIMA":
                    fitted_model, forecast_series = train_evaluate_arima(train_df['Sales'], test_df['Sales'])
                    st.session_state.fitted_model = fitted_model
                    forecast = forecast_series

                elif model_option == "Random Forest":
                    forecast_series = train_evaluate_random_forest(train_df, test_df)
                    features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag7', 'rolling_mean_7']
                    target = 'Sales'
                    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    final_model.fit(data_featured[features], data_featured[target])
                    st.session_state.fitted_model = final_model
                    forecast = forecast_series

                elif model_option == "LSTM":
                    forecast_series = train_evaluate_lstm(train_df['Sales'], test_df['Sales'])
                    st.warning("Note: Future forecasting for LSTM is not implemented in this version.", icon="⚠️")
                    forecast = forecast_series

            st.success("Evaluation forecast generated successfully!")
            st.session_state.forecast_df = pd.DataFrame({'Actual': test_df['Sales'], 'Forecast': forecast})

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
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

        st.sidebar.header("4. Generate Future Forecast")
        future_periods = st.sidebar.number_input("Enter number of days to forecast into the future:", min_value=7, max_value=365, value=30)
        
        if st.sidebar.button("Forecast Future", key="future_forecast"):
            if st.session_state.fitted_model is None:
                st.sidebar.error("Please run an evaluation forecast first (Step 3) to train a model.")
            else:
                with st.spinner("Generating future forecast..."):
                    last_date = data_featured.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods)
                    future_forecast = None
                    
                    if model_option == "ARIMA":
                        future_forecast = st.session_state.fitted_model.forecast(steps=future_periods)

                    elif model_option == "Random Forest":
                        future_df_template = pd.DataFrame(index=future_dates)
                        future_df_template['Sales'] = 0

                        full_data_for_features = pd.concat([data_featured, future_df_template])
                        future_featured_df = create_features(full_data_for_features)
                        
                        features_to_predict = future_featured_df.iloc[-future_periods:]
                        features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag7', 'rolling_mean_7']
                        future_forecast = st.session_state.fitted_model.predict(features_to_predict[features])

                if future_forecast is not None:
                    st.header(f"Future Demand Forecast for the Next {future_periods} Days")
                    future_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
                    future_df.set_index('Date', inplace=True)
                    
                    fig_future = go.Figure()
                    fig_future.add_trace(go.Scatter(x=original_data.index, y=original_data['Sales'], mode='lines', name='Historical Sales'))
                    fig_future.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='Future Forecast', line=dict(color='green')))
                    fig_future.update_layout(title="Future Demand", xaxis_title="Date", yaxis_title="Predicted Sales")
                    st.plotly_chart(fig_future, use_container_width=True)
                    st.dataframe(future_df)
else:
    st.info("Upload a CSV file to begin the forecasting process.")