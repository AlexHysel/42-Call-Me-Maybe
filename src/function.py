from pydantic import BaseModel, PrivateAttr
import json

from src.encoder import Encoder


class Function(BaseModel):
    _name: str = PrivateAttr()
    _t_name: list[int] = PrivateAttr()
    _description: str = PrivateAttr()
    _t_description: list[int] = PrivateAttr()
    _params: dict[str, str] = PrivateAttr()
    _t_params: dict[str, list[int]] = PrivateAttr()
    _t_definition: list[int] = PrivateAttr()

    def __init__(self,
                 function: dict[str, dict[str, str | dict[str, str]]],
                 encoder: Encoder):
        super().__init__()
        self._name = function['name']
        self._t_name = encoder.encode(self._name)
        self._description = function.get('description', '')
        self._t_description = encoder.encode(self._description)
        self._params = {
            k: v['type']
            for k, v in function['parameters'].items()
        }
        self._t_params = {
            k: encoder.encode(v['type'])
            for k, v in function['parameters'].items()
        }
        self._t_definition = encoder.encode(self._to_tool_schema())

    def _to_tool_schema(self) -> str:
        return json.dumps({
            "name": self._name,
            "description": self._description,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": v}
                    for k, v in self._params.items()
                },
                "required": list(self._params.keys())
            }
        })

    @property
    def t_definition(self) -> list[int]:
        return self._t_definition

    @property
    def name(self) -> str:
        return self._name

    @property
    def t_name(self) -> list[int]:
        return self._t_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def t_description(self) -> list[int]:
        return self._t_description

    @property
    def params(self) -> dict[str, str]:
        return self._params

    @property
    def t_params(self) -> dict[str, list[int]]:
        return self._t_params

    @property
    def param_names(self) -> list[str]:
        return list(self._params.keys())
