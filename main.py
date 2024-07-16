import requests
from kivy.app import App
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.video import Video
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout
from win10toast import ToastNotifier
from kivy.uix.checkbox import CheckBox
from kivy.uix.scrollview import ScrollView
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import geocoder
from kivy.graphics import Color, Rectangle
from kivy.graphics import RoundedRectangle
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
import pyttsx3
import speech_recognition as sr
from kivy.uix.popup import Popup



# Weather Alert Code
API_KEY = '118cac35f61293190541b27c80022e80'
ALERT_THRESHOLD_MIN_TEMPERATURE = 19
ALERT_THRESHOLD_MAX_TEMPERATURE = 25
ALERT_PRECIPITATION_THRESHOLD = 5
ALERT_BAD_WEATHER_CONDITION = 'Thunderstorm'
ALERT_WIND_SPEED_THRESHOLD = 2
ALERT_INTERVAL_SECONDS = 60


class WelcomeScreen(RelativeLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define colors
        self.background_color = get_color_from_hex('#201b2e')
        self.text_color = get_color_from_hex('#E8E2DB')
        self.button_color = get_color_from_hex('#4A403D')

        # Background Color
        with self.canvas.before:
            Color(*self.background_color)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Adding the video
        self.video = Video(source='aa.mp4', state='play', options={'allow_stretch': True})
        self.video.bind(texture_size=self._on_video_texture_size)
        self.add_widget(self.video)

        # Adding a semi-transparent background
        self.background = Label(
            size_hint=(1, 0.2),  # 20% of screen height
            pos_hint={'x': 0, 'top': 1}  # Align to the top of the screen
        )
        self.add_widget(self.background)

        # Adding the text label
        '''self.label = Label(
            text="Krishi Parihara",
            font_size=80,
            color=(1, 1, 1, 0.8),  # White color with 70% opacity
            size_hint=(1, 0.2),  # 20% of screen height
            pos_hint={'x': 0, 'top': 1}  # Align to the top of the screen
        )
        self.add_widget(self.label)'''

        # Schedule the switch to HomePage after 30 seconds
        Clock.schedule_once(self.switch_to_home, 15)

    def _on_video_texture_size(self, instance, size):
        aspect_ratio = size[0] / size[1]
        if aspect_ratio > self.width / self.height:
            self.video.size = (self.width, self.width / aspect_ratio)
        else:
            self.video.size = (self.height * aspect_ratio, self.height)

        self.video.size_hint = (1, 1)  # Full screen size_hint
        self.video.pos_hint = {'x': 0, 'y': 0}  # Position at (0, 0) in RelativeLayout

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


    def switch_to_home(self, instance):
        app = App.get_running_app()
        app.root.clear_widgets()
        app.root.add_widget(HomePage())


class WeatherApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # self.city_input = TextInput(hint_text="Enter the city name", multiline=False, halign='center', font_size=30)
        self.city_input = TextInput(
            hint_text="Enter the city name",
            multiline=False,
            halign='center',
            font_size=30,
            padding_y=[(self.height - dp(0)) / 2, 0],  # Vertically center the text
        )
        self.add_widget(self.city_input)

        get_weather_button = Button(text="Get Weather Details", font_size=30)
        get_weather_button.bind(on_press=self.get_weather)
        self.add_widget(get_weather_button)

        self.weather_label = Label(text="", font_size=30)
        self.add_widget(self.weather_label)

        get_forecast_button = Button(text="Get 7-Day Forecast", font_size=30)
        get_forecast_button.bind(on_press=self.get_forecast)
        self.add_widget(get_forecast_button)

        self.forecast_label = Label(text="", font_size=20)
        self.add_widget(self.forecast_label)

    def get_weather(self, instance):
        city = self.city_input.text
        if city:
            try:
                weather_data = self.get_weather_data(city)
                temperature = weather_data['main']['temp'] - 273.15  # Convert to Celsius
                description = weather_data['weather'][0]['description']
                humidity = weather_data['main']['humidity']
                wind_speed = weather_data['wind']['speed']
                weather_text = f"Temperature: {temperature:.2f}°C\nDescription: {description}\nHumidity: {humidity}%\nWind Speed: {wind_speed} m/s"
                self.weather_label.text = weather_text
            except Exception as e:
                self.weather_label.text = f"Failed to get weather data: {e}"
        else:
            self.weather_label.text = "Please enter a city."

    def get_forecast(self, instance):
        city = self.city_input.text
        if city:
            try:
                forecast_data = self.get_forecast_data(city)
                forecast_text = self.format_forecast(forecast_data)
                self.forecast_label.text = forecast_text
            except Exception as e:
                self.forecast_label.text = f"Failed to get forecast data: {e}"
        else:
            self.forecast_label.text = "Please enter a city."

    def get_weather_data(self, city_name):
        api_key = 'fa71156ad750a7986eb6865dd8635596'
        base_url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
        response = requests.get(base_url)
        data = response.json()
        return data

    def get_forecast_data(self, city_name):
        api_key = 'fa71156ad750a7986eb6865dd8635596'
        forecast_url = f'https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}'
        response = requests.get(forecast_url)
        data = response.json()
        return data

    def format_forecast(self, forecast_data):
        forecast_text = "7-Day Weather Forecast:\n"
        previous_date = None

        for forecast in forecast_data['list']:
            date = forecast['dt_txt'].split()[0]  # Extracting date from datetime
            if date != previous_date:  # Only add one entry per day
                temperature = forecast['main']['temp'] - 273.15  # Convert to Celsius
                description = forecast['weather'][0]['description']
                humidity = forecast['main']['humidity']
                wind_speed = forecast['wind']['speed']
                forecast_text += f"{date} - Temp: {temperature:.2f}°C, Description: {description}, Humidity: {humidity}%, Wind Speed: {wind_speed} m/s\n"
                previous_date = date

        return forecast_text


api_key = 'fa71156ad750a7986eb6865dd8635596'
class CropRecommendationApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        self.city_input = TextInput(
            hint_text='Enter the name of the city',
            halign='center',
            font_size=30,
            padding_y=[(self.height - dp(10)) / 2, 0]  # Vertically center the text
        )
        self.recommendation_label = Label(text='')
        self.submit_button = Button(text='Recommend Crop',font_size=30)
        self.submit_button.bind(on_press=self.recommend_crop)

        self.add_widget(self.city_input)
        self.add_widget(self.submit_button)
        self.add_widget(self.recommendation_label)

    def get_weather_data(self, city_name):
        base_url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
        response = requests.get(base_url)
        data = response.json()
        return data

    def load_crop_data(self):
        crop_data = pd.read_csv("crop.csv")  # Update with your CSV file path
        return crop_data

    def recommend_crops(self, temperature, humidity, crop_data):
        # Assuming you have columns like 'N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall', 'label' in your CSV file
        features = crop_data[['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall']]
        target = crop_data['label']

        model = RandomForestClassifier()  # You can use any model of your choice
        model.fit(features, target)

        input_data = [
            [0, 0, 0, temperature, humidity, 0, 0]]  # Replace 0s with actual values for N, P, K, ph, and Rainfall
        crop_prediction = model.predict(input_data)
        return crop_prediction[0]

    def recommend_crop(self, instance):
        city = self.city_input.text
        weather_data = self.get_weather_data(city)
        temperature = weather_data['main']['temp'] - 273.15  # Convert to Celsius
        humidity = weather_data['main']['humidity']

        crop_data = self.load_crop_data()
        crop_recommendation = self.recommend_crops(temperature, humidity, crop_data)

        self.recommendation_label.text = f"Recommended crop for {city}: {crop_recommendation}"
        self.recommendation_label.font_size = '30sp'

class FertilizerRecommendationApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # self.crop_input = TextInput(hint_text='Enter the name of the crop',halign='center', font_size=30)
        self.crop_input = TextInput(
            hint_text='Enter the name of the crop',
            halign='center',
            font_size=30,
            padding_y=[(self.height - dp(10)) / 2, 0]  # Vertically center the text
        )
        # self.place_input = TextInput(hint_text='Enter the name of the place',halign='center', font_size=30)
        self.place_input = TextInput(
            hint_text='Enter the name of the city',
            halign='center',
            font_size=30,
            padding_y=[(self.height - dp(10)) / 2, 0]  # Vertically center the text
        )
        self.recommendation_label = Label(text='')
        self.submit_button = Button(text='Recommend Fertilizer',font_size=30)
        self.submit_button.bind(on_press=self.recommend_fertilizer)

        self.add_widget(self.crop_input)
        self.add_widget(self.place_input)
        self.add_widget(self.submit_button)
        self.add_widget(self.recommendation_label)

    # Function to fetch weather data from OpenWeatherMap API
    def fetch_weather_data(self, city_name):
        api_key = "fa71156ad750a7986eb6865dd8635596"
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
        response = requests.get(complete_url)
        data = response.json()
        if data["cod"] != "404":
            weather_data = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "moisture": data["main"]["humidity"],  # Assuming humidity as moisture for simplicity
            }
            return weather_data
        else:
            return None

    # Function to recommend fertilizer
    def recommend_fertilizer(self, instance):
        crop_name = self.crop_input.text
        place_name = self.place_input.text

        # Fetch weather data
        weather_data = self.fetch_weather_data(place_name)
        if weather_data is None:
            self.recommendation_label.text = "Error: Failed to fetch weather data for the specified place."
            return

        # Load dataset
        df = pd.read_csv("Fertilizer_Prediction.csv")  # Update path as needed

        # Filter dataset for the specified crop
        crop_df = df[df["Crop"] == crop_name]

        # If no data for the specified crop is found, return 'use manure'
        if crop_df.empty:
            self.recommendation_label.text = "Use manure"
            return

        # Extract temperature values from the dataset
        temperature_list = crop_df["Temperature"].tolist()

        # Find the nearest temperature in the dataset
        nearest_temperature = self.find_nearest_temperature(weather_data["temperature"], temperature_list)

        # Filter dataset for the nearest temperature
        nearest_temp_df = crop_df[crop_df["Temperature"] == nearest_temperature]

        # Assuming fertilizer recommendation is based on temperature, humidity, and moisture
        features = nearest_temp_df[["Temperature"]]
        target = nearest_temp_df["Fertilizer"]

        # Train a machine learning model (Naive Bayes)
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(features, target)

        # Predict the most suited fertilizer
        input_data = [[weather_data["temperature"]]]
        fertilizer_prediction = model.predict(input_data)

        self.recommendation_label.text = f"Recommended fertilizer: {fertilizer_prediction[0]}"
        self.recommendation_label.font_size = '30sp'

    # Function to find the nearest temperature in the dataset
    def find_nearest_temperature(self, temperature, temperature_list):
        return min(temperature_list, key=lambda x: abs(x - temperature))


data = pd.read_csv('soil.csv')

# Split the dataset into features (X) and target (y)
X = data[['N', 'K', 'P']]
y = data['Soil']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)






class BackButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = "Back"
        self.size_hint_y = None
        self.height = dp(50)
        self.width = dp(200)
        self.font_size = dp(18)
        self.background_color = get_color_from_hex('#2C3E50')
        self.color = get_color_from_hex('#ECF0F1')
        self.border_radius = (dp(20),)
        self.bind(size=self.update_button)
        self.bind(pos=self.update_button)

    def update_button(self, instance, value):
        self.canvas.before.clear()
        with self.canvas.before:
            RoundedRectangle(pos=self.pos, size=self.size, radius=self.border_radius)

    def on_press(self):
        # Implement your back button logic here
        app = App.get_running_app()
        app.root.clear_widgets()
        app.root.add_widget(HomePage())



class WeatherForecastPage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        # Adding the Back Button
        self.add_widget(WeatherApp())
        self.add_widget(BackButton())


class CropRecommendationPage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        # Adding the Back Button
        self.add_widget(CropRecommendationApp())
        self.add_widget(BackButton())


class FertilizerRecommendationPage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        # Adding the Back Button
        self.add_widget(FertilizerRecommendationApp())
        self.add_widget(BackButton())


def get_current_weather(api_key, city):
    base_url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def get_current_weather_by_coords(api_key, lat, lon):
    base_url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def check_weather_alert(current_weather, min_threshold, max_threshold, precipitation_threshold, bad_weather_condition,
                        wind_speed_threshold):
    alerts = []

    temperature = current_weather['main']['temp']
    precipitation = current_weather['rain']['1h'] if 'rain' in current_weather else 0
    weather_condition = current_weather['weather'][0]['main']
    wind_speed = current_weather['wind']['speed']

    if temperature < min_threshold or temperature > max_threshold:
        alerts.append(f"Temperature is {temperature}°C.")

    if precipitation > precipitation_threshold:
        alerts.append(f"Expecting rain ({precipitation}mm).")

    if weather_condition == bad_weather_condition:
        alerts.append(f"Bad weather conditions expected (e.g., {bad_weather_condition}).")

    if wind_speed > wind_speed_threshold:
        alerts.append(f"Strong winds expected (Speed: {wind_speed} m/s).")

    return alerts


def send_alert(alerts):
    if alerts:
        message = "\n".join(alerts)
        toaster = ToastNotifier()
        toaster.show_toast("Weather Alerts", message, duration=10)


class AlertSystemPage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.alerts = []

        # layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.city_input = TextInput(hint_text='Enter City Name', multiline=False, halign='center', font_size=30)
        self.add_widget(self.city_input)

        self.current_location_checkbox = CheckBox(active=False, size_hint=(None, None), size=(20, 20))
        self.current_location_checkbox.bind(active=self.on_checkbox_active)
        self.add_widget(Label(text='Use Current Location', height=50))
        self.add_widget(self.current_location_checkbox)

        self.alert_label = Label(text='', size_hint_y=None, height=200, valign='top')
        scroll_view = ScrollView()
        scroll_view.add_widget(self.alert_label)
        self.add_widget(scroll_view)

        self.start_button = Button(text='Start Weather Check', font_size=30, on_press=self.start_weather_check)
        self.add_widget(self.start_button)
        self.add_widget(BackButton())

    def on_checkbox_active(self, checkbox, value):
        if value:
            self.city_input.disabled = True
        else:
            self.city_input.disabled = False

    def start_weather_check(self, instance):
        city = self.city_input.text.strip()
        use_current_location = self.current_location_checkbox.active

        if use_current_location:
            latitude, longitude = self.get_user_location()
            self.city_input.text = f"Latitude: {latitude}, Longitude: {longitude}"
            self.alerts = self.check_weather(latitude=latitude, longitude=longitude)
        else:
            if city:
                self.alerts = self.check_weather(city=city)
            else:
                self.alert_label.text = "Please enter a city name."
                return

        self.display_alerts()

    def get_user_location(self):
        g = geocoder.ip('me')
        return g.latlng

    def check_weather(self, city=None, latitude=None, longitude=None):
        if latitude is not None and longitude is not None:
            current_weather = get_current_weather_by_coords(API_KEY, latitude, longitude)
        else:
            current_weather = get_current_weather(API_KEY, city)

        if current_weather:
            return check_weather_alert(
                current_weather, ALERT_THRESHOLD_MIN_TEMPERATURE, ALERT_THRESHOLD_MAX_TEMPERATURE,
                ALERT_PRECIPITATION_THRESHOLD, ALERT_BAD_WEATHER_CONDITION, ALERT_WIND_SPEED_THRESHOLD
            )
        else:
            return []

    def display_alerts(self):
        if self.alerts:
            message = "\n".join(self.alerts)
            self.alert_label.text = message
            send_alert(self.alerts)
        else:
            self.alert_label.text = "No alerts found."





class HomeButton(Button):
    def __init__(self, text='', icon='', **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = 180

        self.text = text
        self.icon = icon

        self.image = Image(source=self.icon, size_hint=(None, None), size=(50, 50))
        #self.add_widget(self.image)


# Initialize the recognizer
r = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define a function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        # Use Google Speech Recognition to transcribe audio
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Please try again.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

# Define a function to speak text aloud
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

class ChatBotIcon(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (250,250)
        self.pos_hint = {'right': 1, 'y': 0}

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.show_chatbot_popup()
            return True
        return super().on_touch_down(touch)

    def show_chatbot_popup(self):
        # Create and display the chatbot popup here
        popup_content = BoxLayout(orientation='vertical')
        self.chat_history = Label(text="Chat history will display here")
        popup_content.add_widget(self.chat_history)
        self.user_input = TextInput(hint_text="Type your message here")
        popup_content.add_widget(self.user_input)
        ask_button = Button(text="Ask")
        popup_content.add_widget(ask_button)
        popup = Popup(title="Chatbot", content=popup_content, size_hint=(None, None), size=(1100, 600))
        ask_button.bind(on_press=lambda instance: self.process_input())
        popup.open()

    def process_input(self):
        input_text = self.user_input.text
        self.user_input.text = ""  # Clear the input field
        if input_text:
            print("You said: " + input_text)
            # Process user input and generate a response
            if "hello" in input_text.lower():
                response_text = "Hello, how can I help you?"
            elif "name" in input_text.lower():
                response_text="My name is chatbot"
            elif "features" in input_text.lower():
                response_text="It have crop recommendation , fertilizer recommendation , weather forecast, soil analysis, alert system."
            elif "how to use application" in input_text.lower():
                response_text="Go to any one option and get requires details."
            elif "for current weather" in input_text.lower():
                response_text="Go to weather forecast option"
            elif "crops" in input_text.lower():
                response_text="Go to crop recommendation option."
            elif "fertilizer" in input_text.lower():
                response_text="Go to fertilizer recommendation option."
            elif "alert" in input_text.lower():
                response_text="Go to alert system option."
            elif "soil analysis" in input_text.lower():
                response_text="Go to soil analysis option."
            elif "connection error" in input_text.lower():
                response_text = "Kindly check your internet connection or try later!"
            elif "bye" in input_text.lower():
                response_text = "See you"
                speak_text(response_text)
                print("Response:"+ response_text)
            else:
                response_text = "I'm sorry, I didn't understand. Could you please repeat?"
            speak_text(response_text)
            print("Response:"+ response_text)
            self.chat_history.text += f"\nUser: {input_text}\nBot: {response_text}"



class HomePage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Create a GridLayout with two columns
        self.inner_box = GridLayout(cols=2, padding=10, spacing=10)
        self.homepageheading_label = Label(text="KRISHI PARIHARA",font_name="MADEAwelierExtraBold.otf",  font_size=50,
                                           size_hint=(None, None), height=80, pos_hint={'center_x': 0.5})
        self.add_widget(self.homepageheading_label)

        # Create a GridLayout with two columns
        self.inner_box = GridLayout(cols=2, padding=20, spacing=10)

        # Weather Button
        self.weather_button = HomeButton(text="Weather Forecast", icon='weather.png', font_size=40)
        self.weather_button.bind(on_press=self.open_weather_forecast)
        self.inner_box.add_widget(Image(source=self.weather_button.icon, size_hint=(0.5, 0.5)))
        self.inner_box.add_widget(self.weather_button)

        # Crop Button
        self.crop_button = HomeButton(text="Crop Recommendation", icon='crop.png', font_size=40)
        self.crop_button.bind(on_press=self.open_crop_recommendation)
        self.inner_box.add_widget(Image(source=self.crop_button.icon, size_hint=(0.5, 0.5)))
        self.inner_box.add_widget(self.crop_button)

        # Fertilizer Button
        self.fertilizer_button = HomeButton(text="Fertilizer Recommendation", icon='fertilizer.png', font_size=40)
        self.fertilizer_button.bind(on_press=self.open_fertilizer_recommendation)
        self.inner_box.add_widget(Image(source=self.fertilizer_button.icon, size_hint=(0.5, 0.5)))
        self.inner_box.add_widget(self.fertilizer_button)

        # Alert Button
        self.alert_button = HomeButton(text="Alert System", icon='alert.png', font_size=40)
        self.alert_button.bind(on_press=self.open_alert_system)
        self.inner_box.add_widget(Image(source=self.alert_button.icon, size_hint=(0.5, 0.5)))
        self.inner_box.add_widget(self.alert_button)



        # Add the GridLayout to the HomePage
        self.chatbot_icon = ChatBotIcon(source="chatbot.png")
        self.chatbot_icon.bind(on_press=self.open_chatbot)
        self.add_widget(self.inner_box)
        self.add_widget(self.chatbot_icon)
        self.chatbot_icon.pos_hint = {'right': 1, 'y': 0}

    def open_weather_forecast(self, instance):
        app = App.get_running_app()
        app.root.clear_widgets()
        app.root.add_widget(WeatherForecastPage())

    def open_crop_recommendation(self, instance):
        app = App.get_running_app()
        app.root.clear_widgets()
        app.root.add_widget(CropRecommendationPage())

    def open_fertilizer_recommendation(self, instance):
        app = App.get_running_app()
        app.root.clear_widgets()
        app.root.add_widget(FertilizerRecommendationPage())

    def open_alert_system(self, instance):
        app = App.get_running_app()
        app.root.clear_widgets()
        app.root.add_widget(AlertSystemPage())


    def open_chatbot(self, instance):
        instance.show_chatbot_popup()




class MyApp(App):
    def build(self):
        # Window.size = (360, 640)
        return WelcomeScreen()


if __name__ == "__main__":
    MyApp().run()

