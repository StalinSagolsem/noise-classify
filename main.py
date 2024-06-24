from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.core.audio import SoundLoader
from kivy.uix.popup import Popup

import librosa
import numpy as np
import tensorflow as tf
load_model = tf.keras.models.load_model

class AudioClassifierApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes = ['children_playing', 'dog_barking', 'drilling', 'jackhammer', 'siren', 'street_music']
        self.model = load_model('MyBestModel.h5')
        self.selected_file = None
        self.sound = None

    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Title
        title_label = Label(text='Audio Classifier', size_hint_y=None, height=50, font_size=24)
        layout.add_widget(title_label)

        # Upload button
        upload_button = Button(text='Select File', size_hint_y=None, height=50)
        upload_button.bind(on_press=self.show_file_chooser)
        layout.add_widget(upload_button)

        # File name label
        self.file_label = Label(text='No file selected', size_hint_y=None, height=30)
        layout.add_widget(self.file_label)

        # Play button
        play_button = Button(text='Play Audio', size_hint_y=None, height=50)
        play_button.bind(on_press=self.play_audio)
        layout.add_widget(play_button)

        # Predict button
        predict_button = Button(text='Predict', size_hint_y=None, height=50)
        predict_button.bind(on_press=self.predict_audio)
        layout.add_widget(predict_button)

        # Result label
        self.result_label = Label(text='Prediction result will appear here')
        layout.add_widget(self.result_label)

        return layout

    def show_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserListView()
        content.add_widget(file_chooser)

        select_button = Button(text='Select', size_hint_y=None, height=50)
        select_button.bind(on_press=lambda x: self.select_file(file_chooser.selection))
        content.add_widget(select_button)

        popup = Popup(title='Choose Audio File', content=content, size_hint=(0.9, 0.9))
        select_button.bind(on_press=popup.dismiss)
        popup.open()

    def select_file(self, selection):
        if selection:
            self.selected_file = selection[0]
            self.file_label.text = f'Selected: {self.selected_file.split("/")[-1]}'
        else:
            self.file_label.text = 'No file selected'

    def play_audio(self, instance):
        if self.selected_file:
            if self.sound:
                self.sound.stop()
            self.sound = SoundLoader.load(self.selected_file)
            if self.sound:
                self.sound.play()
            else:
                self.result_label.text = 'Error playing audio'
        else:
            self.result_label.text = 'Please select an audio file first'

    def predict_audio(self, instance):
        if not self.selected_file:
            self.result_label.text = 'Please select an audio file first'
            return

        mfcc_features = self.features_extractor(self.selected_file)
        if mfcc_features is None:
            self.result_label.text = 'Error processing audio file'
            return

        raw_scores = self.model.predict(np.array([mfcc_features]))
        pred = np.argmax(raw_scores, axis=-1)
        pred_class = self.classes[pred[0]]

        result_text = f'Predicted class: {pred_class}\n\n'
        for i, class_name in enumerate(self.classes):
            score = raw_scores[0][i] * 100
            result_text += f'{class_name}: {score:.2f}%\n'

        self.result_label.text = result_text

    def features_extractor(self, file):
        try:
            audio, sample_rate = librosa.load(file)
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            return mfccs_scaled_features
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None


if __name__ == '__main__':
    AudioClassifierApp().run()