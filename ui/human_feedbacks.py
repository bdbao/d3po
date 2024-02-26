import os
import json
from tkinter import Tk, Label, Button, LEFT, RIGHT, Canvas
from PIL import Image, ImageTk

desired_size_main = (780, 780)  # Adjust the size of the main image
desired_size_thumbnail = (90, 90)  # Adjust the size of thumbnails
desired_window_width = 700
desired_window_height = 350  

class ImageFeedbackApp:
    def __init__(self, root, image_folder, feedback_file):
        self.root = root
        self.image_folder = image_folder
        self.feedback_file = feedback_file

        self.image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0
        self.feedback_dict = {}

        self.load_images()
        self.create_widgets()

    def load_images(self):
        self.images = [Image.open(os.path.join(self.image_folder, image)).resize(desired_size_main, Image.Resampling.LANCZOS) for image in self.image_list]
        self.photo_images = [ImageTk.PhotoImage(image) for image in self.images]

        # Create thumbnails
        self.thumbnail_images = [image.resize(desired_size_thumbnail, Image.Resampling.LANCZOS) for image in self.images]
        self.thumbnail_photo_images = [ImageTk.PhotoImage(image) for image in self.thumbnail_images]

    def create_widgets(self):
        self.label = Label(self.root, text=self.image_list[self.current_index], image=self.photo_images[self.current_index])
        self.label.pack()

        self.canvas = Canvas(self.root, width=desired_window_width, height=desired_window_height)
        self.canvas.pack(side=LEFT, padx=50)

        self.back_button = Button(self.root, text="Back", command=self.previous_image)
        self.back_button.pack(side=LEFT)

        self.like_button = Button(self.root, text="Like", command=lambda: self.record_feedback(0), fg="green")
        self.like_button.pack(side=RIGHT, padx=15)

        self.dislike_button = Button(self.root, text="Dislike", command=lambda: self.record_feedback(-1), fg="red")
        self.dislike_button.pack(side=RIGHT, padx=15)
        
    def record_feedback(self, feedback):
        if 0 <= self.current_index < len(self.image_list):
            image_name = self.image_list[self.current_index]

            self.feedback_dict[image_name] = feedback # update, create new elements
            self.next_image()

    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.image_list):
            self.update_display()
        else:
            self.save_feedback()
            self.root.destroy()

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def update_display(self):
        self.label.configure(text=self.image_list[self.current_index], image=self.photo_images[self.current_index])
        self.show_thumbnails()

    def show_thumbnails(self):
        self.canvas.delete("all")

        y_position = 50

        for i in range(max(0, self.current_index - 1), min(len(self.image_list), self.current_index + 6)):
            thumbnail_image = self.thumbnail_photo_images[i]
            x_position = 100 * (i - self.current_index + 3) - 120

            # Mark the current image
            if i == self.current_index:
                self.canvas.create_rectangle(x_position - 20, y_position - 20, x_position + 80, y_position + 80, outline="blue", width=3)

            # Display file name
            file_name = self.image_list[i]
            self.canvas.create_text(x_position + 30, y_position + 90, anchor="center", text=file_name)

            # Display thumbnail image
            self.canvas.create_image(x_position + 30, y_position + 30, anchor="center", image=thumbnail_image)
            self.canvas.image = thumbnail_image

    def save_feedback(self):
        feedback_data = list(self.feedback_dict.values())
        grouped_feedback = [feedback_data[i:i + 7] for i in range(0, len(feedback_data), 7)]

        with open(self.feedback_file, "w") as f:
            json.dump(grouped_feedback, f, indent=2)

if __name__ == "__main__":
    image_folder = "2024-02-26-09-50-31_140_bdbao\images"
    feedback_file = "2024-02-26-09-50-31_140_bdbao\json\data.json"

    root = Tk()
    app = ImageFeedbackApp(root, image_folder, feedback_file)
    root.mainloop()
