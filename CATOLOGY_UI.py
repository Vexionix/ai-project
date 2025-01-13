import tkinter as tk
from threading import Thread
from tkinter import scrolledtext

from Catology_AI_proj import Catology_AI


class CATOLOGY_UI:
    def __init__(self):
        # Main window
        self.root = tk.Tk()
        self.root.title("CATOLOGY_GUI")
        self.root.geometry("800x800")
        self.root.resizable(width=True, height=True)

        # ScrolledText widget for displaying chat messages
        self.chat_window = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=40)
        self.chat_window.pack(padx=10, pady=10, fill='both', expand=True)
        self.chat_window.config(state=tk.DISABLED)  # Make the chat window non-editable

        # Entry widget for typing messages
        self.message_entry = tk.Entry(self.root, width=80, font=('Arial', 14))  # Larger font
        self.message_entry.pack(padx=10, pady=20, fill='x')  # Increase padding to make it taller

        # Bind Enter key to send_message function
        self.message_entry.bind('<Return>', self.send_message)

        # Button for sending messages (optional, in case you want to keep the button as well)
        self.send_button = tk.Button(self.root, text="Send", width=20, command=self.send_message)
        self.send_button.pack(pady=10)

        self.WORKER_AI = None
        self.thread = None

    def send_message(self, event=None):
        """Function to send a message."""
        message = self.message_entry.get()
        if message.strip():  # Ensure message is not empty
            self.display_message(message)
            self.WORKER_AI.RECEIVE_TEXT(message)
            self.message_entry.delete(0, tk.END)  # Clear the input field
            if message.lower() == "quit":
                self.stop()

        # Prevent the default behavior of pressing Enter (which would normally insert a newline)
        return "break"
    def display_message_AI(self, message):
        """Function to display the sent message in the chat window."""
        self.chat_window.config(state=tk.NORMAL)  # Enable the chat window for editing
        self.chat_window.insert(tk.END, f"AI: {message}\n")  # Insert the message
        self.chat_window.config(state=tk.DISABLED)  # Disable the chat window again
        self.chat_window.yview(tk.END)  # Scroll to the bottom

    def display_message(self, message):
        """Function to display the sent message in the chat window."""
        self.chat_window.config(state=tk.NORMAL)  # Enable the chat window for editing
        self.chat_window.insert(tk.END, f"You: {message}\n")  # Insert the message
        self.chat_window.config(state=tk.DISABLED)  # Disable the chat window again
        self.chat_window.yview(tk.END)  # Scroll to the bottom

    def LAUNCH_AI_WORKER(self):
        self.WORKER_AI = Catology_AI(self)
        self.thread = Thread(target=self.WORKER_AI.PROCESS_TASKS, daemon=True)
        print("GUI: Worker AI starting...")

        self.thread.start()

    def stop(self):
        self.root.destroy() ## this will close the app and the thread worker is daemon so it will stop also:)

    def launch(self):
        """Launch the Tkinter mainloop."""
        self.LAUNCH_AI_WORKER()

        self.root.mainloop()


# Run the application
if __name__ == "__main__":
    app = CATOLOGY_UI()
    app.root.protocol("WM_DELETE_WINDOW", app.stop)
    app.launch()
