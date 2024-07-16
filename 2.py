from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from skimage import io, color, feature
import numpy as np


class SoilTypeDetector(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.file_chooser = FileChooserListView()
        self.file_chooser.path = "./soil_types"

        self.image_widget = Image()
        self.result_label = Label(text='Select an image to classify soil type')

        classify_button = Button(text='Classify')
        classify_button.bind(on_press=self.classify_image)

        layout.add_widget(self.file_chooser)
        layout.add_widget(self.image_widget)
        layout.add_widget(self.result_label)
        layout.add_widget(classify_button)

        return layout

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
            self.image_widget.source = image_path




if __name__ == '__main__':
    SoilTypeDetector().run()

#'D:\\Krishi\\soil_types'