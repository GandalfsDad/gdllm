from pydantic import BaseModel
from abc import ABC

class AbstractConfig(BaseModel, ABC):
    provider: str