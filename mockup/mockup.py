import tkinter as tk
from tkinter import messagebox

class FatigueMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nuovargio Stebėjimo Sistema")
        self.root.geometry("400x300")

        # Heading Label
        self.heading = tk.Label(self.root, text="Pirminio Maketo Sąsaja", font=("Arial", 16))
        self.heading.pack(pady=10)

        # Buttons
        self.record_button = tk.Button(self.root, text="Įrašyti Garsus", font=("Arial", 12), command=self.start_recording)
        self.record_button.pack(pady=10)

        self.analyze_button = tk.Button(self.root, text="Analizuoti Duomenis", font=("Arial", 12), command=self.analyze_data)
        self.analyze_button.pack(pady=10)

        self.classify_button = tk.Button(self.root, text="Klasifikuoti Nuovargį", font=("Arial", 12), command=self.classify_fatigue)
        self.classify_button.pack(pady=10)

        self.feedback_button = tk.Button(self.root, text="Pateikti Grįžtamąjį Ryšį", font=("Arial", 12), command=self.provide_feedback)
        self.feedback_button.pack(pady=10)

        # Quit Button
        self.quit_button = tk.Button(self.root, text="Išeiti", font=("Arial", 12), command=self.root.quit)
        self.quit_button.pack(pady=10)

    def start_recording(self):
        messagebox.showinfo("Įrašymas", "Garsų įrašymas pradėtas!")

    def analyze_data(self):
        messagebox.showinfo("Analizė", "Akustinių požymių analizė atliekama.")

    def classify_fatigue(self):
        messagebox.showinfo("Klasifikacija", "Nuovargio lygis klasifikuojamas.")

    def provide_feedback(self):
        messagebox.showinfo("Grįžtamasis Ryšys", "Grįžtamasis ryšys pateiktas naudotojui.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FatigueMonitoringApp(root)
    root.mainloop()
