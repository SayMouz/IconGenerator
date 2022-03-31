from tkinter import *
import webbrowser

#Personnalisation
class AppGen:
       def __init__(self):
           self.master = Tk()
           self.master.title("IconGenerator App")
           self.master.geometry("720x580")
           self.master.resizable(height = False, width = False)
           self.master.minsize(480, 360)
           self.master.iconbitmap("electric-generator.ico")
           self.master.config(background = "")



           #intialisation_frame
           #self.interface = Frame()


           #composants
           self.create_widgets()

           #empaquetage
           #self.interface.pack(expand = YES)
           #creation d'image
           # width = 512
           # height = 512
           # image = PhotoImage(file="imggenerator.gif").zoom(35).subsample(32)




       #creation_widgets
       def create_widgets(self):
        self.create_title()
        self.create_textzone()
        self.create_generate_button()
        self.create_exit_button()



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
        icon_button = Button(text = "Générer", font = ("Courrier"), bg = "cyan", fg = "black", command = "")
        icon_button.pack(side = "right",padx =  150)

        # Ajouter_le_button_Fermer
       def create_exit_button(self):
        quitter = Button(text = "Quitter", font = ("Courrier"), bg = "cyan", fg = "black", command = "")
        quitter.pack(side = "left", padx = 150)



        def generate_flavicon_icon():
          webbrowser.open_new("")

#afficher
gen = AppGen()
gen.master.mainloop()