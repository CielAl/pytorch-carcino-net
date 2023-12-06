from typing import List, Dict
import os


def group_by_slidename(file_list: List[str], delimiter: str = '_', position: int = 0) -> Dict[str, List[str]]:
    o_dict: Dict[str, List[str]] = dict()
    for file in file_list:
        file_part, _ = os.path.splitext(os.path.basename(file))
        slide_name = file_part.split(delimiter)[position]
        o_dict[slide_name] = o_dict.get(slide_name, [])
        o_dict[slide_name].append(file)
    return o_dict

