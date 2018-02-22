import colorama
from .abstract import AbstractController

class MenuController(AbstractController):
    """ 
        This Controller displays the menu to the user, so they can pick
        their option.
    """
    def display(self):
        print(colorama.Fore.BLUE + "Welcome to Fake News Identification")
        print()
        print()
        print("Options:")
        print("q: Quit")
        print(colorama.Fore.RESET)
        return

