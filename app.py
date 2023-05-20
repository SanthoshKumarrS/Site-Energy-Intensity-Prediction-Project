import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction,ordinal_encoder,string_to_float


# Load the model
model = joblib.load(r'Model/random_forest.joblib')

st.set_page_config(page_title="Site Energy Intensity Prediction",
                   page_icon="🏙🏘", layout="wide")


#creating option list for dropdown menu
options_building_class = ['Commercial', 'Residential'] 

options_facility_type = ['Grocery_store_or_food_market', 'Public_Assembly_Social_meeting','Mixed_Use_Predominantly_Commercial', 'Religious_worship',
                'Office_Medical_non_diagnostic', 'Warehouse_Distribution_or_Shipping_center','Service_Vehicle_service_repair_shop',
                'Warehouse_Selfstorage', 'Commercial_Other', 'Retail_Enclosed_mall', 'Public_Assembly_Movie_Theater',
                'Retail_Strip_shopping_mall', 'Retail_Uncategorized','Education_Other_classroom', 'Office_Uncategorized', 'Food_Sales',
                'Warehouse_Nonrefrigerated', 'Industrial', 'Warehouse_Refrigerated','Education_Uncategorized', 'Data_Center',
                'Office_Bank_or_other_financial','Lodging_Hotel', 'Food_Service_Uncategorized', 'Parking_Garage',
                'Food_Service_Restaurant_or_cafeteria', 'Health_Care_Uncategorized','Service_Uncategorized', 'Education_College_or_university',
                'Public_Assembly_Stadium', 'Laboratory', 'Health_Care_Inpatient','Commercial_Unknown', 'Lodging_Uncategorized',
                'Health_Care_Outpatient_Clinic', 'Public_Assembly_Uncategorized','Public_Assembly_Entertainment_culture', 'Food_Service_Other',
                'Public_Assembly_Drama_theater', 'Retail_Vehicle_dealership_showroom','Nursing_Home', 'Public_Assembly_Recreation',
                'Education_Preschool_or_daycare', 'Mixed_Use_Commercial_and_Residential','Service_Drycleaning_or_Laundry',
                'Multifamily_Uncategorized','Public_Safety_Uncategorized', 'Public_Safety_Fire_or_police_station',
                'Public_Assembly_Library', 'Office_Mixed_use', 'Public_Assembly_Other','Lodging_Dormitory_or_fraternity_sorority',
                'Lodging_Other','Mixed_Use_Predominantly_Residential','Health_Care_Outpatient_Uncategorized', 'Public_Safety_Penitentiary',
                'Public_Safety_Courthouse', '2to4_Unit_Building', 'Warehouse_Uncategorized',
                '5plus_Unit_Building'] 


features = ['energy_star_rating',
            'facility_type',
            'floor_area',
            'year_built',
            'days_with_fog',
            'elevation',
            'building_class',
            'direction_peak_wind_speed',
            'max_wind_speed',
            'direction_max_wind_speed']



st.markdown("<h1 style='text-align: center;'>Site Energy Intensity Prediction App 🏙🏘</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        energy_star_rating = st.number_input("Energy star rating: ", value=0,max_value=100,format="%d")
        facility_type = st.selectbox("Facility type: ", options=options_facility_type)
        floor_area = st.number_input("floor area: ", value=0,max_value=100000, format="%d")
        year_built = st.number_input("Year built: ", value=1900,min_value=1700,max_value=2023, format="%d")
        days_with_fog = st.number_input("Days with Fog: ", value=0,min_value=0,max_value=365, format="%d")
        building_class = st.selectbox("Building Class: ", options=options_building_class)
        direction_peak_wind_speed = st.number_input("Direction of peak wind speed: ", value=0,min_value=0,max_value=360, format="%d")
        max_wind_speed = st.number_input("Max wind speed: ", value=0,min_value=0,max_value=30, format="%d")
        elevation = st.number_input("Elevation: ", value=0,min_value=0,max_value=20, format="%d")
        direction_max_wind_speed = st.number_input("Direction Max wind speed: ", value=0,min_value=0,max_value=360, format="%d")

        submit = st.form_submit_button("Predict")


    if submit:
        facility_type = ordinal_encoder(facility_type, options_facility_type)
        building_class = ordinal_encoder(building_class, options_building_class)
        

        data = np.array([energy_star_rating,facility_type,floor_area,year_built,days_with_fog,elevation,building_class,
                        direction_peak_wind_speed,max_wind_speed,direction_max_wind_speed]).reshape(1,-1)
        print(data)
        pred = get_prediction(data=data, model=model)

        st.write(f"Predicted Site EUI is:  {pred[0]}")




if __name__ == '__main__':
    main()

