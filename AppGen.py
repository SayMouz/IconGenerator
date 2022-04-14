import webbrowser
from tkinter import *
from PIL import Image, ImageTk
import time


#Personnalisation


class AppGen:
    def __init__(self):
       self.master = Tk()
       self.master.title("IconGenerator App")
       self.master.geometry("720x720")
       #self.master.resizable(height = False, width = False)
       self.master.minsize(480, 360)
       self.master.iconbitmap("electric-generator.ico")
       self.master.config(background = "")

       #intialisation_frame
       # self.interface = Frame()
       # self.frame_button = LabelFrame(self)
       # self.frame_button.configure(text='Button Frame')
       # self.frame_button.grid(column=1, row=1)
       # self.frame_button.grid(padx=20, pady=20)

       #composants
       self.create_widgets()

       #empaquetage
       #self.interface.pack(expand = YES)


    #creation_widgets
    def create_widgets(self):
        self.create_title()
        self.create_textzone()
        self.create_generate_button()
        self.create_exit_button()
        self.create_image_zone()
        #self.update_clock()

    # Ajouter la zone pour l'image
    def create_image_zone(self):
        # creation d'image
        load = Image.open("icones_gen.PNG")
        image = load.resize((250, 250), Image.ANTIALIAS)
        # print(photo.size, photo.width, photo.height)
        photo = ImageTk.PhotoImage(image)
        label_img = Label(self.master, image=photo)
        label_img.place(x=230, y=250)
        # load.show()

        """
        canvas = Canvas(width=350, height=200)
        canvas.create_image(50, 50, image=photo)
        canvas.pack()
        """
        # canvas.grid(row=1, column=4, sticky=N)

   #Ajouter_un_titre
    def create_title(self):
        label_title = Label(text = "Icon Generator APP", font = ("Century", 40))
        label_title.pack()

   #Ajouter_une_zone_de_saisie
    def create_textzone(self):
        saisie = Entry()
        saisie.pack(padx = 100, pady = 100)

    #Ajouter_le_button
    def create_generate_button (self):
        icon_button = Button(self.master,text = "Générer", font = ("Courrier"), bg = "cyan", fg = "black", command = "")
        icon_button.pack(side = "right",padx =  150)

    # Ajouter_le_button_Fermer
    def create_exit_button(self):
        quitter = Button(text = "Quitter", font = ("Courrier"), bg = "cyan", fg = "black", command = "destroy .")
        quitter.pack(side = "left", padx = 150)
        # self.button_quit = Button(self.frame_button)
        # self.button_quit.config(text='Quit')
        # self.button_quit.grid(column=2, row=1)
        # self.button_quit.configure(command=self.quit_app)

        # def update_clock(self):
        #  now = time.strftime("%H:%M:%S")
        #  self.label.configure(text=now)
        #  self.root.after(1000, self.update_clock)

    def generate_flavicon_icon(self):
        webbrowser.open_new("")

#afficher
gen = AppGen()
gen.master.mainloop()