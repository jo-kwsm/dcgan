import dataclasses
import pprint
from typing import Any, Dict, Tuple
import yaml

__all__ = ["get_config"]


@dataclasses.dataclass
class Config:
    model: str

    batch_size: int = 64
    num_workers: int = 16
    max_epoch: int = 1000

    g_lr: float = 0.0001
    d_lr: float = 0.0004
    beta1: float = 0.0
    beta2: float = 0.9

    size: int = 64
    z_dim: int = 20

    train_csv: str = "./csv/data.csv"

    def __post_init__(self) -> None:
        self._type_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _type_check(self) -> None:
        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            if hasattr(field_type, "__origin__"):
                element_type = field_type.__args__[0]
                field_type = field_type.__origin__

                self._type_check_element

            if type(_dict[field]) is not field_type:
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )


    def _type_check_element(
        self, field: str, vals: Tuple[Any], element_type: type
    ) -> None:
        for val in vals:
            if type(val) is not element_type:
                raise TypeError(
                    f"The element of '{field}' field is supposed to be {element_type}."
                )


def convert_list2tuple(_dict: Dict[str, Any]):
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict


def get_config(config_path: str):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuple(config_dict)
    config = Config(**config_dict)
    return config
