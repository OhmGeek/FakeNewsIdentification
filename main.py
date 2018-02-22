import colorama
from abc import ABC, abstractmethod
# THIS FILE SHOULD PRESENT OUTPUT TO BENCHMARK ABOVE QUESTIONS.

class AbstractController(ABC):
    @abstractmethod
    def display(self):
        pass

class MenuController(AbstractController):
    def display(self):
        print(colorama.Fore.BLUE + "Welcome to Fake News Identification")
        print()
        print()
        print("Options:")
        print("q: Quit")
        print(colorama.Fore.RESET)
        return

class QuitController(AbstractController):
    def display(self):
        print("Bye!")
        exit(0)


def get_controller_instance(text_input):
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