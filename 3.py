from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from skimage import io, color, feature
import os


class SoilTypeDetector(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.file_chooser = FileChooserIconView(filters=['*.png', '*.jpg'])
        self.file_chooser.path = "./"

        self.image_widget = Image()
        self.result_label = Label(text='Select an image to classify soil type')

        classify_button = Button(text='Classify')
        classify_button.bind(on_press=self.classify_image)

        layout.add_widget(self.file_chooser)

        # Scroll view for displaying thumbnails
        thumbnails_layout = GridLayout(cols=4, spacing=10, size_hint_y=None)
        thumbnails_layout.bind(minimum_height=thumbnails_layout.setter('height'))
        thumbnails_scrollview = ScrollView(size_hint=(1, 0.8))
        thumbnails_scrollview.add_widget(thumbnails_layout)
        layout.add_widget(thumbnails_scrollview)

        layout.add_widget(self.image_widget)
        layout.add_widget(self.result_label)
        layout.add_widget(classify_button)

        # Show thumbnails
        self.update_thumbnails()

        return layout

    def update_thumbnails(self):
        # Clear existing thumbnails
        self.file_chooser._update_files()

        thumbnails_layout = self.file_chooser._icon_layout
        thumbnails_layout.clear_widgets()

        # Show thumbnails of images in the selected folder
        for filename in os.listdir(self.file_chooser.path):
            if filename.lower().endswith(('.png', '.jpg')):
                thumbnail = Image(source=os.path.join(self.file_chooser.path, filename), size_hint_y=None, height=100)
                thumbnails_layout.add_widget(thumbnail)

    def load_model(self):
        # Load your pretrained model here
        # model = ...
        # return model

        # For demonstration, returning a dummy model
        return DummyModel()

    def classify_image(self, instance):
        # Get the selected image path from the file chooser
        image_path = self.file_chooser.selection and self.file_chooser.selection[0]

        if image_path:
            # Load and preprocess the selected image
            image = io.imread(image_path)
            gray_image = color.rgb2gray(image)
            hog_features = feature.hog(gray_image).reshape(1, -1)

            # Load the pretrained model
            model = self.load_model()

            # Classify the image
            predicted_soil_type = model.predict(hog_features)[0]

            self.result_label.text = f'Predicted soil type: {predicted_soil_type}'

            # Display the selected image
            self.image_widget.source = 'D:\\Krishi\\soil_types'


class DummyModel:
    def predict(self, features):
        # Dummy prediction, replace this with your actual model prediction
        return ['Sandy']  # Dummy output


if __name__ == '__main__':
    SoilTypeDetector().run()
