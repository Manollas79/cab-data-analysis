import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import math
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor   # <-- Added Decision Tree import
from sklearn.cluster import KMeans

# ------------------------------
# Custom CSS for Styling
# ------------------------------
st.markdown(
    """
    <style>
    body { background-color: #f0f2f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stSidebar { background-color: #cccccc; }
    .sidebar .sidebar-content { background-color: #cccccc; padding: 20px; }
    .css-18e3th9 { padding: 1rem 1rem 10rem 1rem; }
    div.stButton > button { background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer; }
    div.stButton > button:hover { background-color: #45a049; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Database Setup (Not used in this version)
# ------------------------------
engine = create_engine('sqlite:///rides_data.db', echo=False)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class Ride(Base):
    __tablename__ = 'rides'
    id = Column(Integer, primary_key=True)
    ride_id = Column(String)
    fare_amount = Column(Float)
    pickup_datetime = Column(DateTime)
    passenger_count = Column(Integer)
    pickup_longitude = Column(Float)
    pickup_latitude = Column(Float)
    dropoff_longitude = Column(Float)
    dropoff_latitude = Column(Float)
    distance = Column(Float)

Base.metadata.create_all(engine)

# ------------------------------
# Haversine Function (for old dataset)
# ------------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371 * c

# ------------------------------
# Functions to Load Datasets
# ------------------------------
@st.cache_data(show_spinner=True)
def load_old_dataset(nrows):
    try:
        df_old = pd.read_csv("trips_data.csv", nrows=nrows, parse_dates=["tpep_pickup_datetime"], low_memory=True)
        st.info("Loaded old dataset from trips_data.csv")
    except FileNotFoundError:
        st.warning("trips_data.csv not found. Using simulated old dataset.")
        num_samples = nrows
        ride_times = pd.date_range(start='2023-01-01', periods=num_samples, freq='h')
        df_old = pd.DataFrame({
            "VendorID": np.random.choice([1,2], num_samples),
            "tpep_pickup_datetime": ride_times,
            "tpep_dropoff_datetime": ride_times + pd.to_timedelta(np.random.randint(5,30, num_samples), unit='m'),
            "passenger_count": np.random.randint(1, 5, num_samples),
            "trip_distance": np.random.uniform(0.5, 10, num_samples).round(2),
            "pickup_longitude": np.random.uniform(77.5, 77.7, num_samples),
            "pickup_latitude": np.random.uniform(12.8, 13.0, num_samples),
            "RateCodeID": np.random.randint(1, 6, num_samples),
            "store_and_fwd_flag": np.random.choice(['Y', 'N'], num_samples),
            "dropoff_longitude": np.random.uniform(77.5, 77.7, num_samples),
            "dropoff_latitude": np.random.uniform(12.8, 13.0, num_samples),
            "fare_amount": np.random.uniform(5, 50, num_samples).round(2)
        })
    df_old.rename(columns={
        "tpep_pickup_datetime": "ride_time",
        "fare_amount": "price"
    }, inplace=True)
    if "ride_id" not in df_old.columns:
        df_old["ride_id"] = df_old.index.astype(str)
    df_old["distance"] = df_old["trip_distance"]
    df_old.dropna(subset=["ride_time", "passenger_count", "pickup_longitude", "pickup_latitude",
                          "dropoff_longitude", "dropoff_latitude", "distance"], inplace=True)
    df_old["price"] = df_old["price"].clip(lower=0)
    df_old["distance"] = haversine(df_old["pickup_longitude"], df_old["pickup_latitude"],
                                   df_old["dropoff_longitude"], df_old["dropoff_latitude"])
    # If 'area_name' column is not present, create a placeholder
    if "area_name" not in df_old.columns:
        df_old["area_name"] = "Unknown"
    df_old["dataset_type"] = "old"
    return df_old

@st.cache_data(show_spinner=True)
def load_new_dataset(nrows):
    try:
        df_new = pd.read_csv("trips_fare.csv", nrows=nrows, low_memory=True)
        st.info("Loaded new dataset from trips_fare.csv")
    except FileNotFoundError:
        st.warning("trips_fare.csv not found. Using simulated new dataset.")
        num_samples = nrows
        df_new = pd.DataFrame({
            "trip_duration": np.random.randint(5, 120, num_samples),
            "distance_traveled": np.random.uniform(1, 20, num_samples).round(2),
            "num_of_passengers": np.random.randint(1, 5, num_samples),
            "fare": np.random.uniform(5, 50, num_samples).round(2),
            "tip": np.random.uniform(0, 10, num_samples).round(2),
            "miscellaneous_fees": np.random.uniform(0, 5, num_samples).round(2),
            "total_fare": np.random.uniform(10, 70, num_samples).round(2),
            "surge_applied": np.random.uniform(1.0, 3.0, num_samples).round(1)
        })
    df_new.rename(columns={
        "distance_traveled": "distance",
        "num_of_passengers": "passenger_count"
    }, inplace=True)
    df_new["price"] = df_new["total_fare"]
    df_new.dropna(subset=["trip_duration", "distance", "passenger_count", "price", "surge_applied"], inplace=True)
    df_new["dataset_type"] = "new"
    df_new["ride_time"] = pd.to_datetime("2023-01-01")  # dummy value
    if "ride_id" not in df_new.columns:
        df_new["ride_id"] = df_new.index.astype(str)
    return df_new

# ------------------------------
# Load Datasets
# ------------------------------
max_rows = st.sidebar.number_input("Max Rows to Load", min_value=1000, max_value=1000000,
                                     value=10000, step=1000, key="max_rows")
df_old = load_old_dataset(nrows=max_rows)
df_new = load_new_dataset(nrows=max_rows)
df_price = df_new.copy()  # For Price Analysis

# ------------------------------
# Cached Price Prediction Model Training for New Dataset
# ------------------------------
@st.cache_data(show_spinner=True)
def train_price_prediction_model_new(df):
    features = df[["distance", "passenger_count", "trip_duration"]]
    target = df["price"]
    df_model = df.dropna(subset=features.columns.tolist() + ["price"])
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return model, mse

model_new, mse_new = train_price_prediction_model_new(df_price)

# ------------------------------
# Advanced Models for Price Analysis (with Decision Tree)
# ------------------------------
@st.cache_data(show_spinner=True)
def train_advanced_models(df):
    features = df[["distance", "passenger_count", "trip_duration"]]
    target = df["price"]
    df_model = df.dropna(subset=features.columns.tolist() + ["price"])
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_mse = mean_squared_error(y_test, lin_model.predict(X_test))
    
    dt_model = DecisionTreeRegressor(random_state=42)  # <-- Decision Tree model added
    dt_model.fit(X_train, y_train)
    dt_mse = mean_squared_error(y_test, dt_model.predict(X_test))
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_mse = mean_squared_error(y_test, rf_model.predict(X_test))
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_mse = mean_squared_error(y_test, gb_model.predict(X_test))
    
    models = {
        "Linear Regression": (lin_model, lin_mse),
        "Decision Tree": (dt_model, dt_mse),
        "Random Forest": (rf_model, rf_mse),
        "Gradient Boosting": (gb_model, gb_mse)
    }
    return models

advanced_models = train_advanced_models(df_price)

# ------------------------------
# Updated area_coords dictionary with provided coordinates
# ------------------------------
area_coords = {
    "Silk Board Junction": (12.9177, 77.6233),
    "Marathahalli Bridge": (12.9560, 77.7010),
    "Dairy Circle": (12.9346, 77.6101),
    "Koramangala BDA Complex": (12.9279, 77.6271),
    "Madiwala Market": (12.9172, 77.6226),
    "Central Silk Board": (12.9177, 77.6233),
    "Vidyaranyapura": (13.0722, 77.5596),
    "KR Puram": (13.0097, 77.6970),
    "Whitefield": (12.9698, 77.7499),
    "Hebbal Flyover": (13.0524, 77.5915),
    "Mysore Road": (12.9538, 77.5150),
    "Yesvantpur Underbridge": (13.0285, 77.5308),
    "Bannerghatta Road": (12.8879, 77.6006),
    "JP Nagar 5th Phase 24th Main": (12.9072, 77.5855),
    "KR Market Junction": (12.9636, 77.5776),
    "Banashankari Bus Stand Junction": (12.9416, 77.5649),
    "M.G. Road": (12.9756, 77.6067),
    "Brigade Road": (12.9742, 77.6090),
    "Church Street": (12.9752, 77.6059),
    "Commercial Street": (12.9817, 77.6087),
    "Rajajinagar": (12.9916, 77.5547),
    "J.P. Nagar": (12.9000, 77.5850),
    "HAL Signal": (12.9616, 77.6445),
    "Trinity Circle": (12.9738, 77.6200),
    "Navarang Signal": (12.9935, 77.5540),
    "Marathahalli Signal": (12.9560, 77.7010),
    "Murgesh Palya Signal": (12.9616, 77.6445),
    "Domlur Signal": (12.9616, 77.6400),
    "KR Circle": (12.9719, 77.5946)
}

# ------------------------------
# Define function to plot Top 20 Hotspots by area using area_coords and frequency count from area_name
# ------------------------------
def plot_hotspots_by_area(df, area_coords):
    if "area_name" not in df.columns:
        st.error("Column 'area_name' not found in dataset. Cannot plot hotspots.")
        return

    # Compute frequency for each area in df
    freq = df["area_name"].value_counts().head(20)
    # Create a folium map centered on Bengaluru
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
    
    # Loop through the top 20 areas and add a marker if coordinates exist in area_coords
    for area, count in freq.items():
        if area in area_coords:
            lat, lon = area_coords[area]
            folium.Marker(
                location=[lat, lon],
                popup=f"{area}<br>Rides: {count}",
                icon=folium.Icon(color="blue", icon="map-marker", prefix="fa")
            ).add_to(m)
    st_folium(m, width=700, height=500)

# ------------------------------
# Sidebar Navigation (Pages)
# ------------------------------
st.title("Uber Rides Data Analysis App")
available_pages = ["Dashboard", "Trips Analysis", "Dataset", "Price Analysis"]
page = st.sidebar.radio("Select a Feature", available_pages)

# ------------------------------
# Dataset Page
# ------------------------------
if page == "Dataset":
    st.header("Dataset")
    st.write("**Old Dataset (for analysis):**")
    st.dataframe(df_old)
    st.write("**New Dataset (for price prediction):**")
    st.dataframe(df_new)
    csv_old = df_old.to_csv(index=False).encode("utf-8")
    csv_new = df_new.to_csv(index=False).encode("utf-8")
    st.download_button("Download Old Dataset CSV", data=csv_old, file_name="trips_data.csv", mime="text/csv")
    st.download_button("Download New Dataset CSV", data=csv_new, file_name="trips_fare.csv", mime="text/csv")

# ------------------------------
# Dashboard Page – Metrics from New Dataset; Visualizations from Old Dataset
# ------------------------------
if page == "Dashboard":
    st.header("Dashboard")
    total_trips_new = len(df_new)
    avg_fare_new = df_new["price"].mean()
    nonzero_fares_new = df_new[df_new["price"] > 0]["price"]
    min_fare_new = nonzero_fares_new[nonzero_fares_new > 35].min() if not nonzero_fares_new[nonzero_fares_new > 35].empty else 35
    max_fare_new = df_new["price"].max()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trips", total_trips_new)
    col2.metric("Average Fare", f"₹ {avg_fare_new:.2f}")
    col3.metric("Min Fare", f"₹ {min_fare_new:.2f}")
    col4.metric("Max Fare", f"₹ {max_fare_new:.2f}")
    
    st.subheader("Basic Visualizations (Using Old Dataset)")
    viz_option = st.selectbox("Select Basic Visualization", 
                              ["Average Fare by Hour", "Daily Trips Over Time", "Fare Distribution", "Trips vs Fare (Scatter)", "Weekday vs Weekend Fare Trends"], key="dashboard_viz")
    if viz_option == "Average Fare by Hour":
        df_old["hour"] = pd.to_datetime(df_old["ride_time"], dayfirst=True).dt.hour
        data = df_old.groupby("hour")["price"].mean().reset_index()
        title = "Average Fare by Hour"
        x_col = "hour"
        y_col = "price"
        chart_options = ["Bar Chart", "Line Chart", "Raw Data"]
    elif viz_option == "Daily Trips Over Time":
        df_old["date"] = pd.to_datetime(df_old["ride_time"], dayfirst=True).dt.date
        data = df_old.groupby("date").size().reset_index(name="trips")
        title = "Daily Trips Over Time"
        x_col = "date"
        y_col = "trips"
        chart_options = ["Line Chart", "Raw Data"]
    elif viz_option == "Fare Distribution":
        data = df_old[["price"]]
        title = "Fare Distribution"
        x_col = "price"
        y_col = None
        chart_options = ["Bar Chart", "Pie Chart", "Raw Data"]
    elif viz_option == "Trips vs Fare (Scatter)":
        data = df_old[["distance", "price"]]
        title = "Trips vs Fare: Distance vs Fare"
        x_col = "distance"
        y_col = "price"
        chart_options = ["Scatter Chart", "Raw Data"]
    elif viz_option == "Weekday vs Weekend Fare Trends":
        df_old["day_of_week"] = pd.to_datetime(df_old["ride_time"], dayfirst=True).dt.day_name()
        df_old["day_type"] = df_old["day_of_week"].apply(lambda x: "Weekend" if x in ["Saturday", "Sunday"] else "Weekday")
        data = df_old.groupby("day_type")["price"].mean().reset_index()
        title = "Average Fare: Weekday vs Weekend"
        x_col = "day_type"
        y_col = "price"
        chart_options = ["Bar Chart", "Pie Chart", "Raw Data"]
    
    chart_type = st.selectbox("Select Chart Type", chart_options, key="dashboard_chart")
    if chart_type == "Raw Data":
        st.dataframe(data)
    else:
        if viz_option == "Fare Distribution" and chart_type == "Pie Chart":
            data['Fare Range'] = pd.cut(data[x_col], bins=10).astype(str)
            pie_data = data['Fare Range'].value_counts().reset_index()
            pie_data.columns = ['Fare Range', 'Count']
            fig = px.pie(pie_data, names='Fare Range', values='Count', title=title)
        else:
            if chart_type == "Bar Chart":
                fig = px.bar(data, x=x_col, y=y_col, title=title)
            elif chart_type == "Line Chart":
                fig = px.line(data, x=x_col, y=y_col, title=title)
            elif chart_type == "Pie Chart":
                fig = px.pie(data, names=x_col, values=y_col, title=title)
            elif chart_type == "Scatter Chart":
                fig = px.scatter(data, x=x_col, y=y_col, title=title)
            else:
                fig = px.bar(data, x=x_col, y=y_col, title=title)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 20 Hotspots in Bengaluru (Using Old Dataset)")
    plot_hotspots_by_area(df_old, area_coords)

# ------------------------------
# Price Analysis Page (New Dataset Only) with Advanced Model Options
# ------------------------------
if page == "Price Analysis":
    st.header("Price Analysis")
    st.subheader("Select Model for Fare Estimation")
    model_choice = st.selectbox("Model", list(advanced_models.keys()), key="model_choice")
    chosen_model, chosen_mse = advanced_models[model_choice]
    st.write(f"{model_choice} Mean Squared Error: {chosen_mse:.2f}")
    
    st.subheader("Enter Details for Estimation")
    col1, col2, col3 = st.columns(3)
    with col1:
        pp_distance = st.number_input("Distance Traveled (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1, key="pp_distance_new")
    with col2:
        pp_passenger = st.number_input("Number of Passengers", min_value=1, max_value=7, value=1, step=1, key="pp_passenger_new")
    with col3:
        pp_duration = st.number_input("Trip Duration (minutes)", min_value=5, max_value=180, value=30, step=1, key="pp_duration_new")
    
    day_of_week = st.selectbox("Select Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="pp_day")
    hour_12 = st.selectbox("Select Hour (1-12)", list(range(1,13)), key="pp_hour")
    am_pm = st.radio("AM/PM", ["AM", "PM"], key="pp_ampm")
    
    if am_pm == "AM":
        converted_hour = 0 if hour_12 == 12 else hour_12
    else:
        converted_hour = hour_12 if hour_12 == 12 else hour_12 + 12
    
    new_input = pd.DataFrame({
        "distance": [pp_distance],
        "passenger_count": [pp_passenger],
        "trip_duration": [pp_duration]
    })
    
    # If Decision Tree is selected, adjust prediction if needed
    if model_choice == "Decision Tree":
        dt_pred = chosen_model.predict(new_input)[0]
        reg_pred = advanced_models["Linear Regression"][0].predict(new_input)[0]
        rf_pred = advanced_models["Random Forest"][0].predict(new_input)[0]
        avg_val = (reg_pred + rf_pred) / 2
        if abs(dt_pred - avg_val) > 20:
            base_prediction = avg_val
        else:
            base_prediction = dt_pred
    else:
        base_prediction = chosen_model.predict(new_input)[0]
    
    passenger_multiplier = 1.0 if pp_passenger <= 5 else 1.2
    extra_duration = max(pp_duration - 10, 0)
    duration_multiplier = 1 + 0.1 * (extra_duration // 10)
    weekend_multiplier = 1.1 if day_of_week in ["Saturday", "Sunday"] else 1.0
    peak_multiplier = 1.1 if (8 <= converted_hour <= 10 or 16 <= converted_hour <= 18) else 1.0
    midnight_multiplier = 1.1 if (converted_hour >= 23 or converted_hour < 5) else 1.0
    
    final_fare = base_prediction * passenger_multiplier * duration_multiplier * weekend_multiplier * peak_multiplier * midnight_multiplier
    st.success(f"Estimated Fare: ₹ {final_fare:.2f}")
    
    if "pp_predictions" not in st.session_state:
        st.session_state.pp_predictions = pd.DataFrame(columns=[
        "distance", "passenger_count", "trip_duration", "day_of_week",
        "hour", "am_pm", "predicted_fare", "predicted_model"  # Ensure this column exists
    ])

if st.button("Save Estimation", key="pp_save_prediction"):
    new_row = pd.DataFrame([{
        "distance": pp_distance,
        "passenger_count": pp_passenger,
        "trip_duration": pp_duration,
        "day_of_week": day_of_week,
        "hour": converted_hour,
        "am_pm": am_pm,
        "predicted_fare": final_fare,
        "predicted_model": model_choice  # Ensure this is added correctly
    }])

    # Update session state DataFrame
    st.session_state.pp_predictions = pd.concat([st.session_state.pp_predictions, new_row], ignore_index=True)
    
    st.success("Estimation saved!")

# Display the saved estimations
if not st.session_state.pp_predictions.empty:
    st.write("Saved Estimations:")
    st.dataframe(st.session_state.pp_predictions)

    # Convert to CSV
    csv_predictions = st.session_state.pp_predictions.to_csv(index=False).encode("utf-8")

    # Download button
    st.download_button("Download Estimations CSV", data=csv_predictions, file_name="estimations.csv", mime="text/csv")

# ------------------------------
# Trips Analysis Page – Only Additional Visualizations (Using Old Dataset)
# ------------------------------
if page == "Trips Analysis":
    st.header("Trips Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filters")
        passenger_options = sorted(df_old["passenger_count"].unique())
        passengers = st.multiselect("Select Passenger Count", options=passenger_options, default=passenger_options)
    with col2:
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01").date(), key="start_date_old")
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31").date(), key="end_date_old")
    
    filtered_df = df_old[
        (df_old["passenger_count"].isin(passengers)) &
        (pd.to_datetime(df_old["ride_time"], dayfirst=True).dt.date >= start_date) &
        (pd.to_datetime(df_old["ride_time"], dayfirst=True).dt.date <= end_date)
    ]
    
    if filtered_df.empty:
        st.warning("No trips found for the selected filters.")
    else:
        st.subheader("Additional Visualizations")
        add_viz_options = st.multiselect("Select Additional Visualizations", [
            "Fare Trends Over Time",
            "Daily Demand of Cabs",
            "Daywise Demand of Cabs",
            "Hourly Demand of Car",
            "Top 20 Hotspots Map",
            "Time vs Fare",
            "Clustered Hotspots"
        ], key="additional_viz")
        
        for viz in add_viz_options:
            if viz == "Fare Trends Over Time":
                st.subheader("Fare Trends Over Time")
                filtered_df["ride_date"] = pd.to_datetime(filtered_df["ride_time"], dayfirst=True).dt.date
                data = filtered_df.groupby("ride_date")["price"].mean().reset_index(name="avg_fare")
                title = "Fare Trends Over Time"
                x_col = "ride_date"
                y_col = "avg_fare"
                chart_type = st.selectbox("Select Chart Type for Fare Trends", ["Line Chart", "Bar Chart", "Raw Data"], key="trend_add")
                if chart_type == "Raw Data":
                    st.dataframe(data)
                elif chart_type == "Line Chart":
                    fig_add = px.line(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
                elif chart_type == "Bar Chart":
                    fig_add = px.bar(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
                    
            elif viz == "Daily Demand of Cabs":
                st.subheader("Daily Demand of Cabs")
                filtered_df["date"] = pd.to_datetime(filtered_df["ride_time"], dayfirst=True).dt.date
                data = filtered_df.groupby("date").size().reset_index(name="bookings")
                title = "Daily Demand of Cabs"
                x_col = "date"
                y_col = "bookings"
                chart_type = st.selectbox("Select Chart Type for Daily Demand", ["Bar Chart", "Line Chart", "Raw Data"], key="daily_add")
                if chart_type == "Raw Data":
                    st.dataframe(data)
                elif chart_type == "Bar Chart":
                    fig_add = px.bar(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
                elif chart_type == "Line Chart":
                    fig_add = px.line(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
            elif viz == "Daywise Demand of Cabs":
                st.subheader("Daywise Demand of Cabs")
                filtered_df["day_name"] = pd.to_datetime(filtered_df["ride_time"], dayfirst=True).dt.day_name()
                data = filtered_df.groupby("day_name").size().reset_index(name="bookings")
                title = "Daywise Demand of Cabs"
                x_col = "day_name"
                y_col = "bookings"
                chart_type = st.selectbox("Select Chart Type for Daywise Demand", ["Bar Chart", "Pie Chart", "Raw Data"], key="daywise_add")
                if chart_type == "Raw Data":
                    st.dataframe(data)
                elif chart_type == "Bar Chart":
                    fig_add = px.bar(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
                elif chart_type == "Pie Chart":
                    fig_add = px.pie(data, names=x_col, values=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
            elif viz == "Hourly Demand of Car":
                st.subheader("Hourly Demand of Cabs")
                filtered_df["hour"] = pd.to_datetime(filtered_df["ride_time"], dayfirst=True).dt.hour
                data = filtered_df.groupby("hour").size().reset_index(name="bookings")
                title = "Hourly Demand of Cabs"
                x_col = "hour"
                y_col = "bookings"
                chart_type = st.selectbox("Select Chart Type for Hourly Demand", ["Bar Chart", "Line Chart", "Raw Data"], key="hourly_add")
                if chart_type == "Raw Data":
                    st.dataframe(data)
                elif chart_type == "Bar Chart":
                    fig_add = px.bar(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
                elif chart_type == "Line Chart":
                    fig_add = px.line(data, x=x_col, y=y_col, title=title)
                    st.plotly_chart(fig_add, use_container_width=True)
            elif viz == "Top 20 Hotspots Map":
                st.subheader("Top 20 Hotspots in Bengaluru")
                plot_hotspots_by_area(filtered_df, area_coords)
            elif viz == "Time vs Fare":
                st.subheader("Time vs Fare")
                if "ride_time" in filtered_df.columns and "price" in filtered_df.columns:
                    data = filtered_df.sort_values("ride_time")[["ride_time", "price"]].set_index("ride_time")
                    chart_option = st.selectbox("Select Chart Type for Time vs Fare", ["Bar Chart", "Line Chart", "Raw Data"], key="time_vs_fare_chart")
                    if chart_option == "Raw Data":
                        st.dataframe(data)
                    elif chart_option == "Line Chart":
                        st.line_chart(data)
                    elif chart_option == "Bar Chart":
                        data_reset = data.reset_index()
                        fig = px.bar(data_reset, x="ride_time", y="price", title="Time vs Fare")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Required columns not found for Time vs Fare Chart.")
            elif viz == "Clustered Hotspots":
                st.subheader("Clustered Hotspots (K-Means)")
                # Use area_coords if area_name exists; else fallback to lat/lon columns
                if "area_name" in filtered_df.columns:
                    coords_df = filtered_df.copy()
                    coords_df["lat"] = coords_df["area_name"].apply(lambda x: area_coords[x][0] if x in area_coords else np.nan)
                    coords_df["lon"] = coords_df["area_name"].apply(lambda x: area_coords[x][1] if x in area_coords else np.nan)
                    coords_df = coords_df.dropna(subset=["lat", "lon"])
                else:
                    coords_df = filtered_df[["pickup_latitude", "pickup_longitude"]].dropna()
                    coords_df = coords_df.rename(columns={"pickup_latitude": "lat", "pickup_longitude": "lon"})
                if not coords_df.empty:
                    n_clusters = st.number_input("Number of Clusters", min_value=3, max_value=20, value=10, step=1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(coords_df[["lat", "lon"]])
                    coords_df["cluster"] = clusters
                    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=["lat", "lon"])
                    m_clusters = folium.Map(location=[coords_df["lat"].mean(), coords_df["lon"].mean()], zoom_start=11)
                    for i, row in centroids.iterrows():
                        folium.Marker(location=[row["lat"], row["lon"]],
                                      popup=f"Cluster {i}",
                                      icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_clusters)
                    st_folium(m_clusters, width=700, height=500)
                else:
                    st.write("Not enough coordinate data for clustering.")
