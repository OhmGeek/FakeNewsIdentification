import colorama

from controllers.abstract import AbstractController
from controllers.menu import MenuController
from controllers.quit import QuitController

def get_controller_instance(text_input):
    """ This takes a string, and outputs an instance of the specified Controller """
    processed_text_input = text_input.lower().strip()
    options = {
        'q': QuitController(),
    }
    return options.get(processed_text_input)
def main():
    is_running = True
    controller = None
    while is_running:
        controller = MenuController()
        controller.display()

        text_input = input("Option: ")
        
        controller = get_controller_instance(text_input)
        controller.display()


if __name__ == '__main__':
    main()