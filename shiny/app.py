import faicons as fa
from shiny import App, reactive, render, ui
from datetime import datetime
import requests
from geopy.distance import geodesic
import pandas as pd
from datetime import timedelta
import joblib
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget 
from ipyleaflet import Map, Marker, Polyline
import numpy as np
from shiny.ui import div, HTML
from shinyswatch import theme
import pytz

# Get Model
lgb_url = "https://raw.githubusercontent.com/meitiandouzai/STAT-628-Model3/refs/heads/main/shiny/lightgbm_model.joblib"
reg_url = "https://raw.githubusercontent.com/meitiandouzai/STAT-628-Model3/refs/heads/main/shiny/feature_coefficients(2).csv"

airports = [
        "ABE", "ABI", "ABQ", "ABR", "ABY", "ACT", "ACV", "ACY", "ADK", "ADQ", "AEX", "AGS", "ALB", "ALO", "ALW", 
        "AMA", "ANC", "APN", "ART", "ASE", "ATL", "ATW", "AUS", "AVL", "AVP", "AZA", "AZO", "BDL", "BET", "BFF", 
        "BFL", "BGM", "BGR", "BHM", "BIL", "BIS", "BJI", "BKG", "BLI", "BLV", "BMI", "BNA", "BOI", "BOS", "BPT", 
        "BQK", "BQN", "BRD", "BRO", "BRW", "BTM", "BTR", "BTV", "BUF", "BUR", "BWI", "BZN", "CAE", "CAK", "CDC", 
        "CDV", "CGI", "CHA", "CHO", "CHS", "CID", "CIU", "CKB", "CLE", "CLL", "CLT", "CMH", "CMI", "CMX", "CNY", 
        "COD", "COS", "COU", "CPR", "CRP", "CRW", "CSG", "CVG", "CWA", "CYS", "DAB", "DAL", "DAY", "DBQ", "DCA", 
        "DEN", "DFW", "DHN", "DIK", "DLH", "DRO", "DRT", "DSM", "DTW", "DUT", "DVL", "EAR", "EAT", "EAU", "ECP", 
        "EGE", "EKO", "ELM", "ELP", "ERI", "ESC", "EUG", "EVV", "EWN", "EWR", "EYW", "FAI", "FAR", "FAT", "FAY", 
        "FCA", "FLG", "FLL", "FLO", "FNT", "FSD", "FSM", "FWA", "GCC", "GCK", "GEG", "GFK", "GGG", "GJT", "GNV", 
        "GPT", "GRB", "GRI", "GRK", "GRR", "GSO", "GSP", "GTF", "GTR", "GUC", "GUM", "HDN", "HGR", "HHH", "HIB", 
        "HLN", "HNL", "HOB", "HOU", "HPN", "HRL", "HSV", "HTS", "HVN", "HYS", "IAD", "IAG", "IAH", "ICT", "IDA", 
        "IFP", "ILM", "IMT", "IND", "INL", "IPT", "ISN", "ISP", "ITH", "ITO", "JAC", "JAN", "JAX", "JFK", "JHM", 
        "JLN", "JMS", "JNU", "KOA", "KTN", "LAN", "LAR", "LAS", "LAW", "LAX", "LBB", "LBE", "LBF", "LBL", "LCH", 
        "LCK", "LEX", "LFT", "LGA", "LGB", "LIH", "LIT", "LNK", "LNY", "LRD", "LSE", "LWB", "LWS", "LYH", "MAF", 
        "MBS", "MCI", "MCO", "MDT", "MDW", "MEI", "MEM", "MFE", "MFR", "MGM", "MHK", "MHT", "MIA", "MKE", "MKG", 
        "MKK", "MLB", "MLI", "MLU", "MMH", "MOB", "MOT", "MQT", "MRY", "MSN", "MSO", "MSP", "MSY", "MTJ", "MYR", 
        "OAJ", "OAK", "OGD", "OGG", "OGS", "OKC", "OMA", "OME", "ONT", "ORD", "ORF", "ORH", "OTH", "OTZ", "OWB", 
        "PAH", "PBG", "PBI", "PDX", "PGD", "PGV", "PHF", "PHL", "PHX", "PIA", "PIB", "PIE", "PIH", "PIT", "PLN", 
        "PNS", "PPG", "PQI", "PRC", "PSC", "PSE", "PSG", "PSM", "PSP", "PUB", "PUW", "PVD", "PVU", "PWM", "RAP", 
        "RDD", "RDM", "RDU", "RFD", "RHI", "RIC", "RKS", "RNO", "ROA", "ROC", "ROP", "ROW", "RST", "RSW", "SAF", 
        "SAN", "SAT", "SAV", "SBA", "SBN", "SBP", "SBY", "SCC", "SCE", "SCK", "SDF", "SEA", "SFB", "SFO", "SGF", 
        "SGU", "SHD", "SHV", "SIT", "SJC", "SJT", "SJU", "SLC", "SLN", "SMF", "SMX", "SNA", "SPI", "SPN", "SPS", 
        "SRQ", "STC", "STL", "STS", "STT", "STX", "SUN", "SUX", "SWF", "SWO", "SYR", "TLH", "TOL", "TPA", "TRI", 
        "TTN", "TUL", "TUS", "TVC", "TWF", "TXK", "TYR", "TYS", "UIN", "USA", "VEL", "VLD", "VPS", "WRG", "XNA", 
        "YAK", "YKM", "YNG", "YUM"
    ]

operations = [
                "9E", "9K", "AA", "AS", "AX", "B6", "C5", "CP", "DL", 
                "EM", "EV", "F9", "G4", "G7", "HA", "KS", "MQ", "NK", 
                "OH", "OO", "PT", "QX", "UA", "VX", "WN", "YV", "YX", "ZW"
            ]
# Load models
response_model = requests.get(lgb_url)
with open("lightgbm_model.joblib", "wb") as f:
    f.write(response_model.content)
response_coeff = requests.get(reg_url)
with open("feature_coefficients.csv", "wb") as f:
    f.write(response_coeff.content)
model_delay = joblib.load('lightgbm_model.joblib')
model_cancel = pd.read_csv("feature_coefficients.csv")

# get present time
# US central time
central_time_zone = pytz.timezone('US/Central')
now1 = datetime.now(central_time_zone)
now = now1.replace(tzinfo=None)
current_hour = str(now.hour)
current_minute = f"{now.minute:02d}"
now_plus_one_hour = now1 + timedelta(hours=1)
arrival_hour = str(now_plus_one_hour.hour)
arrival_minute = f"{now_plus_one_hour.minute:02d}" 

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize("departure", "Departure:", choices=airports, selected='ORD'),
        ui.input_selectize("arrival", "Arrival:", choices=airports),
        ui.input_selectize("operating", "Your Operating Airline:", choices=operations),
        ui.input_date("date", "Your departure date:"),
        ui.input_selectize("hour", "Expected departure time (hour):", [str(i) for i in range(24)], selected=current_hour),
        ui.input_selectize("minute", "Expected departure time (minute):", [f"{i:02d}" for i in range(60)], selected=current_minute),

        ui.input_date("date2", "Your arrival date:"),
        ui.input_selectize("hour2", "Expected arrival time (hour):", [str(i) for i in range(24)], selected=arrival_hour),
        ui.input_selectize("minute2", "Expected arrival time (minute):", [f"{i:02d}" for i in range(60)], selected=arrival_minute),
        ui.div(
            ui.input_action_button("submit", "Submit"),
            style="display: flex; justify-content: center; align-items: center; height: 100%;"
            ),
        open="always"
    ),
    ui.page_fluid(
        # Make the main content scrollable
        output_widget("map"),
        ui.value_box(
            showcase=fa.icon_svg("clock"), 
            title="Predicted arrival delay time (min)",
            value=ui.output_text("show_model_delay")
        ),
        ui.value_box(
            showcase=fa.icon_svg("plane"),
            title="Predicted arrival cancel probability (%)",
            value=ui.output_text("show_model_cancel")
        ),

        ui.output_ui("hints"),

        # Add scrollable content area
        div(
            HTML("""
                <p><b>Designed by Group 4</p>
                <p><b>Contact Info: </b><u>rma235@wisc.edu</u></p>
                <p><b>Contributor:</b> Yuchen Xu, Mario Ma, Yiteng Tu, Yudi Wang</p>
            """),
            style="text-align: center; margin-top: 30px; font-size: 12px;"
        ),
    ),
    title="Flight Delay&Cancel Prediction",
    fillable=True,
    style="height: 100vh;",
    theme=theme.darkly
)


def server(input, output, session):
    api_key1 = "ykMIQt8uoQ6nA/d58xvvGg==qgPhf2js4eTDhaN8" #airport location API
    api_key2 = "3ef2f57ac16845feb7b161004241011"  # weather API


    # Get coordinates for the city name
    def get_coordinates(airport_ID):
        api_url = f'https://api.api-ninjas.com/v1/airports?iata={airport_ID}'
        response = requests.get(api_url, headers={'X-Api-Key': f'{api_key1}'})
        if response.status_code == requests.codes.ok:
                data = response.json()
                if data:  # Check if there's any data in the response
                    latitude = data[0].get('latitude')
                    longitude = data[0].get('longitude')
                    return round(float(latitude), 2), round(float(longitude), 2)
                else:
                    return 1,1
        else:
            return 1,1


    def get_time_block(hour, minute):
        time_str = f"{int(hour):02d}{int(minute):02d}"
        time_blocks = {
            ('0000', '0559'): "0001-0559",
            ('0600', '0659'): "0600-0659",
            ('0700', '0759'): "0700-0759",
            ('0800', '0859'): "0800-0859",
            ('0900', '0959'): "0900-0959",
            ('1000', '1059'): "1000-1059",
            ('1100', '1159'): "1100-1159",
            ('1200', '1259'): "1200-1259",
            ('1300', '1359'): "1300-1359",
            ('1400', '1459'): "1400-1459",
            ('1500', '1559'): "1500-1559",
            ('1600', '1659'): "1600-1659",
            ('1700', '1759'): "1700-1759",
            ('1800', '1859'): "1800-1859",
            ('1900', '1959'): "1900-1959",
            ('2000', '2059'): "2000-2059",
            ('2100', '2159'): "2100-2159",
            ('2200', '2259'): "2200-2259",
            ('2300', '2359'): "2300-2359",
        }
        for time_range, block in time_blocks.items():
            if time_str >= time_range[0] and time_str <= time_range[1]:
                return block
        return "Unknown Time Block"

    def get_distance(dep_coords, arr_coords):
        if not dep_coords or not arr_coords:
            return np.nan
        distance = geodesic(dep_coords, arr_coords).miles
        return distance

    # Get weather data for the given coordinates
    def get_weather(lat, lon, date, hour):
        location = f"{lat}, {lon}"
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key2}&q={location}&days=5&dt={date}&hour={hour}&aqi=no&alerts=no"
        response = requests.get(url)
        if response.status_code == 200:
            weather_data = response.json()
            forecast_day = weather_data["forecast"]["forecastday"][0]
            forecast_hour = forecast_day["hour"]
            
            # Find the specific weather
            for hour_data in forecast_hour:
                if hour_data["time"] == f"{date} {hour}:00":
                    temperature = hour_data["temp_c"]
                    wind_speed = hour_data["wind_mph"]* 0.44704
                    wind_direction = round(hour_data["wind_degree"] / 10) * 10
                    pressure = hour_data["pressure_in"]* 33.8639
                    precipitation = hour_data["precip_in"]
                    humidity = hour_data["humidity"]
                    dew_point = hour_data["dewpoint_c"]
                    visibility = hour_data["vis_km"]
                    gust_speed = hour_data["gust_mph"]* 0.44704

                    # tidy data
                    return {
                        "temperature": temperature,
                        "wind_speed": wind_speed,
                        "wind_direction": wind_direction,
                        "pressure": pressure,
                        "precipitation": precipitation,
                        "humidity": humidity,
                        "dew_point": dew_point,
                        "visibility": visibility,
                        "gust_speed": gust_speed
                    }
            return 1
        else:
            return 1

    def map_weather_to_model_input(dep_weather_data, arr_weather_data, input_data=None):
        dep_hour = int(input_data.get('hour', 0))
        dep_minute = int(input_data.get('minute', 0))
        arr_hour = int(input_data.get('hour2', 0))
        arr_minute = int(input_data.get('minute2', 0))

        dep_time_block = get_time_block(dep_hour, dep_minute)
        arr_time_block = get_time_block(arr_hour, arr_minute)
        
        def categorize_wind_direction(wind_direction):
            if wind_direction < 0 or wind_direction >= 360:
                return np.nan
            
            if 0 <= wind_direction < 40:
                return '0-40'
            elif 50 <= wind_direction <= 90:
                return '50-90'
            elif 90 < wind_direction <= 130:
                return '90-130'
            elif 140 <= wind_direction <= 180:
                return '140-180'
            elif 180 < wind_direction <= 220:
                return '180-220'
            elif 230 <= wind_direction <= 270:
                return '230-270'
            elif 270 < wind_direction <= 310:
                return '270-310'
            elif 320 <= wind_direction <= 350:
                return '320-350'
            else:
                return np.nan


        def get_weather_info(weather_data):
            return {
                "HourlyAltimeterSetting": weather_data.get("pressure_in", np.nan),  
                "HourlyDewPointTemperature": weather_data.get("dew_point", np.nan),  
                "HourlyDryBulbTemperature": weather_data.get("temperature", np.nan),  
                "HourlyPrecipitation": weather_data.get("precipitation", np.nan), 
                "HourlyRelativeHumidity": weather_data.get("humidity", np.nan),  
                "HourlySeaLevelPressure": weather_data.get("pressure", np.nan), 
                "HourlyVisibility": weather_data.get("visibility", np.nan),  
                "HourlyWindDirection": categorize_wind_direction(weather_data.get("wind_direction", np.nan)),
                "HourlyWindGustSpeed": weather_data.get("gust_speed", np.nan), 
                "HourlyWindSpeed": weather_data.get("wind_speed", np.nan), 
            }
        
        date = input_data.get('date', '') 
        year = str(date.year) if date else np.nan
        month = str(date.month) if date else np.nan
        day_of_week = str(date.weekday() + 1) if date else np.nan

        
        def categorize_holiday(date):
            if not date:
                return np.nan
            
            # get year
            year = date.year
            begin = datetime(year, 11, 1)  # just consider vacation
            # find thanksgiving
            thanksgiving = begin + timedelta(days=(3 - begin.weekday()) + 21) 
            christmas = datetime(year, 12, 25)
            newyear = datetime(year, 1, 1)

            if isinstance(date, datetime):
                date = date.date()  # change to datetime.date
            
            # compare date
            if begin.date() <= date < thanksgiving.date() - timedelta(days=1):
                return 'Before Thanksgiving'
            elif thanksgiving.date() - timedelta(days=1) <= date <= thanksgiving.date() + timedelta(days=4):
                return 'Around Thanksgiving'
            elif thanksgiving.date() + timedelta(days=4) < date < christmas.date() - timedelta(days=1):
                return 'Between Thanksgiving and Christmas'
            elif christmas.date() - timedelta(days=1) <= date <= christmas.date() + timedelta(days=1):
                return 'Around Christmas'
            elif christmas.date() + timedelta(days=1) < date < newyear.date() - timedelta(days=1):
                return 'Between Christmas and Newyear'
            elif newyear.date() - timedelta(days=1) <= date <= newyear.date() + timedelta(days=1):
                return 'Around Newyear'
            else:
                return 'After Newyear'

        # holiday category
        holiday = categorize_holiday(date)

        operating_airline = input.operating()
        origin_city = input.departure()
        dest_city = input.arrival()
        
        dep_coords = get_coordinates(origin_city) 
        arr_coords = get_coordinates(dest_city) 
        distance = get_distance(dep_coords, arr_coords)
        
        # get weather
        origin_weather_info = get_weather_info(dep_weather_data) if dep_weather_data != "Unable to retrieve weather data." else {}
        dest_weather_info = get_weather_info(arr_weather_data) if arr_weather_data != "Unable to retrieve weather data." else {}
        
        # input data
        model_input = {
            "Year": year,
            "Month": month,
            "DayOfWeek": day_of_week,
            "Operating_Airline ": operating_airline,
            "Origin": origin_city,
            "Dest": dest_city,
            "DepTimeBlk": dep_time_block,
            "ArrTimeBlk": arr_time_block,
            "Distance": distance, 
            **{f"Origin_{key}": value for key, value in origin_weather_info.items()},  
            **{f"Dest_{key}": value for key, value in dest_weather_info.items()},
            "Holiday": holiday, 
        }
        
        return model_input


    # --------------------------output

    @output
    @render_widget  
    def map():
        departure_airport = input.departure()
        arrival_airport = input.arrival()

        # Get coordinates for departure and arrival points
        dep_coords = get_coordinates(departure_airport)
        arr_coords = get_coordinates(arrival_airport)
        mymap = Map(center=(39.50, -98.35), zoom=4) 
        point_dep = Marker(location=dep_coords, draggable=False, title=f"Departure: {departure_airport}")
        point_arr = Marker(location=arr_coords, draggable=False, title=f"Arrival: {arrival_airport}" ) 
        line = Polyline(
                locations=[dep_coords, arr_coords],
                color="blue",
                weight=3,
                opacity=0.8
            )
        mymap.add_layer(point_dep)
        mymap.add_layer(point_arr)
        mymap.add_layer(line)
        return mymap

    @reactive.calc
    @reactive.event(input.submit)
    def model_delay_output():
        dep = input.departure()
        arr = input.arrival()
        date = input.date()
        hour = int(input.hour()) + 1 if int(input.minute()) >= 30 else int(input.hour())
        hour = f"{hour:02d}"
        minute = input.minute()
        date2 = input.date2()
        hour2 = int(input.hour2()) + 1 if int(input.minute2()) >= 30 else int(input.hour2())
        hour2 = f"{hour2:02d}"
        minute2 = input.minute2()

        # check place (can't be same)
        if dep == arr:
            return "Please choose different places."
        
        dep_time = pd.Timestamp(f"{date} {hour}:{minute}")
        arr_time = pd.Timestamp(f"{date2} {hour2}:{minute2}")

        if dep_time < now - pd.Timedelta(minutes=2):
            return "Your departure time should be later than present."
        elif arr_time <= dep_time:
            return "Your arrival time should be later than departure time."
        elif arr_time - dep_time > pd.Timedelta(hours=20):
            return "The flight can't be such a long time."
        elif arr_time - now > pd.Timedelta(days=5):
            return "The arrival time is out of weather prediction range."
    
        # Get weather for both departure and arrival
        dep_lat, dep_lon = get_coordinates(dep)
        arr_lat, arr_lon = get_coordinates(arr)

        # check if the data of the airport is captured by API
        if dep_lat == 1 or dep_lon == 1:
            return "No airport data found in Origin."
        if arr_lat == 1 or arr_lon == 1:
            return "No airport data found in Destination."

        dep_weather = get_weather(dep_lat, dep_lon, date, hour)
        arr_weather = get_weather(arr_lat, arr_lon, date2, hour2)

        if dep_weather == 1:
            return "Weather data for the origin not found."
    
        if arr_weather == 1:
            return "Weather data for the destination not found."
        
        model_input = map_weather_to_model_input(dep_weather, arr_weather, {
            'date': date,
            'departure': dep,
            'arrival': arr,
            'hour': hour,
            'minute': minute,
            'hour2': hour2,
            'minute2': minute2,
        })

        # change to dataframe
        data1 = pd.DataFrame([model_input])

        categorical_columns = [
            'Year', 'Month', 'DayOfWeek', 'Operating_Airline ', 'Origin', 'Dest', 
            'DepTimeBlk', 'ArrTimeBlk', 'Origin_HourlyWindDirection', 
            'Dest_HourlyWindDirection', 'Holiday'
        ]
        data1[categorical_columns] = data1[categorical_columns].astype('category')

        # float
        float_cols = [
            'Origin_HourlyAltimeterSetting', 'Origin_HourlyDewPointTemperature', 
            'Origin_HourlyDryBulbTemperature', 'Origin_HourlyPrecipitation', 'Origin_HourlyRelativeHumidity', 
            'Origin_HourlySeaLevelPressure', 'Origin_HourlyVisibility', 'Origin_HourlyWindGustSpeed', 
            'Origin_HourlyWindSpeed', 'Dest_HourlyAltimeterSetting', 'Dest_HourlyDewPointTemperature', 
            'Dest_HourlyDryBulbTemperature', 'Dest_HourlyPrecipitation', 'Dest_HourlyRelativeHumidity', 
            'Dest_HourlySeaLevelPressure', 'Dest_HourlyVisibility', 'Dest_HourlyWindGustSpeed', 
            'Dest_HourlyWindSpeed'
        ]
        data1[float_cols] = data1[float_cols].astype('float64')

        # tidy
        data1['Distance'] = data1['Distance'].astype('int64')

        prediction = model_delay.predict(data1.iloc[[0]])

        min_delay = -253.0
        # from LogArrDelay to ArrDelay
        y_pred_delay = np.expm1(prediction[0]) + min_delay - 1

        return round(y_pred_delay,2)  

    @output
    @render.text
    @reactive.event(input.submit)
    def show_model_delay():
        return model_delay_output()

    @reactive.calc
    @reactive.event(input.submit)
    def model_cancel_output():
        dep = input.departure()
        arr = input.arrival()
        date = input.date()
        hour = int(input.hour()) + 1 if int(input.minute()) >= 30 else int(input.hour())
        hour = f"{hour:02d}"
        minute = input.minute()
        date2 = input.date2()
        hour2 = int(input.hour2()) + 1 if int(input.minute2()) >= 30 else int(input.hour2())
        hour2 = f"{hour2:02d}"
        minute2 = input.minute2()

        # check place (can't be same)
        if dep == arr:
            return "Please choose different places."
        
        dep_time = pd.Timestamp(f"{date} {hour}:{minute}")
        arr_time = pd.Timestamp(f"{date2} {hour2}:{minute2}")

        if dep_time < now - pd.Timedelta(minutes=2):
            return "Your departure time should be later than present."
        elif arr_time <= dep_time:
            return "Your arrival time should be later than departure time."
        elif arr_time - dep_time > pd.Timedelta(hours=20):
            return "The flight can't be such a long time."
        elif arr_time - now > pd.Timedelta(days=5):
            return "The arrival time is out of weather prediction range."
    
        # Get weather for both departure and arrival
        dep_lat, dep_lon = get_coordinates(dep)
        arr_lat, arr_lon = get_coordinates(arr)

        # check if the data of the airport is captured by API
        if dep_lat == 1 or dep_lon == 1:
            return "No airport data found in Origin."
        if arr_lat == 1 or arr_lon == 1:
            return "No airport data found in Destination."

        dep_weather = get_weather(dep_lat, dep_lon, date, hour)
        arr_weather = get_weather(arr_lat, arr_lon, date2, hour2)

        if dep_weather == 1:
            return "Weather data for the origin not found."
    
        if arr_weather == 1:
            return "Weather data for the destination not found."
        
        model_input = map_weather_to_model_input(dep_weather, arr_weather, {
            'date': date,
            'departure': dep,
            'arrival': arr,
            'hour': hour,
            'minute': minute,
            'hour2': hour2,
            'minute2': minute2,
        })

        data1 = pd.DataFrame([model_input])

        # define type
        categorical_columns = [
            'DayOfWeek', 'Operating_Airline ', 'Origin', 'Dest', 
            'DepTimeBlk', 'ArrTimeBlk', 'Origin_HourlyWindDirection', 
            'Dest_HourlyWindDirection', 'Holiday'
        ]
        float_cols = [
            'Origin_HourlyDewPointTemperature', 'Origin_HourlyDryBulbTemperature', 
            'Origin_HourlyPrecipitation', 'Origin_HourlyRelativeHumidity', 
            'Origin_HourlySeaLevelPressure', 'Origin_HourlyVisibility', 
            'Origin_HourlyWindGustSpeed', 'Origin_HourlyWindSpeed', 
            'Dest_HourlyDewPointTemperature', 'Dest_HourlyDryBulbTemperature', 
            'Dest_HourlyPrecipitation', 'Dest_HourlyRelativeHumidity', 
            'Dest_HourlySeaLevelPressure', 'Dest_HourlyVisibility', 
            'Dest_HourlyWindGustSpeed', 'Dest_HourlyWindSpeed'
        ]
        integer_column = 'Distance'

        # category to dummies
        data1[categorical_columns] = data1[categorical_columns].astype('category')
        data1 = pd.get_dummies(data1, columns=categorical_columns, drop_first=False)

        # float and int
        data1[float_cols] = data1[float_cols].astype('float64')
        data1[integer_column] = data1[integer_column].astype('int64')

        # delete
        data1 = data1.drop(columns=["Year", "Month", "Origin_HourlyAltimeterSetting", "Dest_HourlyAltimeterSetting"])

        # make dictionary
        coefficients = dict(zip(model_cancel['Feature'], model_cancel['Coefficient']))

        # missing features should be 0
        missing_features = set(data1.columns) - set(coefficients.keys())
        for feature in missing_features:
            coefficients[feature] = 0

        means = {
            "Distance": 792.806247,
            "Origin_HourlyDewPointTemperature": 2.310253,
            "Origin_HourlyDryBulbTemperature": 9.656388,
            "Origin_HourlyPrecipitation": 0.065760,
            "Origin_HourlyRelativeHumidity": 64.367087,
            "Origin_HourlySeaLevelPressure": 1018.829208,
            "Origin_HourlyVisibility": 14.818562,
            "Origin_HourlyWindGustSpeed": 1.251675,
            "Origin_HourlyWindSpeed": 3.714047,
            "Dest_HourlyDewPointTemperature": 2.355665,
            "Dest_HourlyDryBulbTemperature": 9.695904,
            "Dest_HourlyPrecipitation": 0.064597,
            "Dest_HourlyRelativeHumidity": 64.464989,
            "Dest_HourlySeaLevelPressure": 1018.995373,
            "Dest_HourlyVisibility": 14.777778,
            "Dest_HourlyWindGustSpeed": 1.231474,
            "Dest_HourlyWindSpeed": 3.700726
        }

        stds = {
            "Distance": 581.028003,
            "Origin_HourlyDewPointTemperature": 9.126297,
            "Origin_HourlyDryBulbTemperature": 8.691776,
            "Origin_HourlyPrecipitation": 0.636459,
            "Origin_HourlyRelativeHumidity": 20.146135,
            "Origin_HourlySeaLevelPressure": 7.364735,
            "Origin_HourlyVisibility": 3.322827,
            "Origin_HourlyWindGustSpeed": 3.749216,
            "Origin_HourlyWindSpeed": 2.417816,
            "Dest_HourlyDewPointTemperature": 9.071661,
            "Dest_HourlyDryBulbTemperature": 8.709794,
            "Dest_HourlyPrecipitation": 0.554202,
            "Dest_HourlyRelativeHumidity": 20.166211,
            "Dest_HourlySeaLevelPressure": 7.183282,
            "Dest_HourlyVisibility": 3.240717,
            "Dest_HourlyWindGustSpeed": 3.717897,
            "Dest_HourlyWindSpeed": 2.403314
        }

        data_with_coeff = data1.apply(
        lambda x: (x - means.get(x.name, 0)) / stds.get(x.name, 1) * coefficients.get(x.name, 0)
        )

        # get prediction
        z = data_with_coeff.sum(axis=1) 
        prediction_probabilities = 1 / (1 + np.exp(-z)) 

        single_prediction = prediction_probabilities.item() * 100
        return f"{single_prediction:.2f}"

    @output
    @render.text
    @reactive.event(input.submit)
    def show_model_cancel():
        return model_cancel_output()

    @output
    @render.text
    @reactive.event(input.submit)
    def hints():
        # Check delay output and try to convert to float
        try:
            delay_time = float(model_delay_output())
        except ValueError:
            delay_time = None

        if delay_time is not None:
            date2 = input.date2()
            hour2 = input.hour2()
            minute2 = input.minute2()
            arr_time = pd.Timestamp(f"{date2} {hour2}:{minute2}") + pd.Timedelta(minutes=delay_time)
        else:
            return None

        # Check cancel output and try to convert to float
        try:
            cancel_probability = float(model_cancel_output())
        except ValueError:
            cancel_probability = None

        if cancel_probability is None:
            return None

        if cancel_probability > 50:
            return "Cancel probability is high, please plan your trip properly."
        elif cancel_probability <= 50:
            if delay_time <= 0:
                return "The flight will arrive on time or earlier than expected."
            elif delay_time > 0:
                return HTML(f'<span style="font-size: 18px; font-weight: bold;">The expected arrival time will be delayed until {arr_time.strftime("%Y-%m-%d %H:%M:%S")}.\
                            </span><br><span style="font-size: 18px; font-weight: bold;">The probability of flight cancellation is low.</span>')

app = App(app_ui, server)

