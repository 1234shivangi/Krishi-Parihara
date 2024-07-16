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
        self.video = Video(source='logo.mp4', state='play', options={'allow_stretch': True})
        self.video.bind(texture_size=self._on_video_texture_size)
        self.add_widget(self.video)

        # Adding a semi-transparent background
        self.background = Label(
            size_hint=(1, 0.2),  # 20% of screen height
            pos_hint={'x': 0, 'top': 1}  # Align to the top of the screen
        )
        self.add_widget(self.background)

        # Schedule the switch to HomePage after 30 seconds
        Clock.schedule_once(self.switch_to_home, 5)

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
