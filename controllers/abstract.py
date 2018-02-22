from abc import ABC, abstractmethod
class AbstractController(ABC):
    """ 
        This is an abstract class representing a controller,
        allowing simple yet efficient interfaces to the user.
    """
    @abstractmethod
    def display(self):
        pass