from Antropic.Usage import Usage


from typing import List

from Antropic.Content import Content
from Antropic.ErrorDetail import ErrorDetail


class AnthropicsApiResponse:
    def __init__(self, content: List['Content'], id: str, model: str, role: str, stop_reason: str, type: str, usage: 'Usage', error: 'ErrorDetail'):
        self.content = content
        self.id = id
        self.model = model
        self.role = role
        self.stop_reason = stop_reason
        self.type = type
        self.usage = usage
        self.error = error