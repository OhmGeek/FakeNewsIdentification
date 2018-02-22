from .abstract import AbstractController

class QuitController(AbstractController):
    """ This controller simply exits gracefully """
    def display(self):
        print("Bye!")
        exit(0)
