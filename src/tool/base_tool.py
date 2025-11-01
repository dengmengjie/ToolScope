from abc import ABC, abstractmethod

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def short_description(self) -> str:
        return self.description

    @property
    @abstractmethod
    def example(self) -> str:
        pass

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass
