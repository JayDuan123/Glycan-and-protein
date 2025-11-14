import re
import os
import torch
import pickle as pkl
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)

try:
    from glycowork.motif import tokenization
    GLYCOWORK_AVAILABLE = True
except ImportError:
    GLYCOWORK_AVAILABLE = False
    print("Warning: glycowork not available. Using simplified parsing.")


class Glycan:
    def __init__(self,
                 edge_list: Optional[List[List[int]]] = None,
                 unit_type: Optional[torch.Tensor] = None,
                 link_type: Optional[torch.Tensor] = None,
                 glycoword_type: Optional[torch.Tensor] = None,
                 unit_feature: Optional[torch.Tensor] = None,
                 link_feature: Optional[torch.Tensor] = None,
                 glycan_feature: Optional[torch.Tensor] = None,
                 vocab_manager=None,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.vocab_manager = vocab_manager
        self.edge_list = torch.tensor(edge_list, device=self.device) if edge_list else torch.empty((0, 3), dtype=torch.long, device=self.device)
        self.unit_type = unit_type.to(self.device) if unit_type is not None else None
        self.link_type = link_type.to(self.device) if link_type is not None else None
        self.glycoword_type = glycoword_type.to(self.device) if glycoword_type is not None else None
        self.unit_feature = unit_feature.to(self.device) if unit_feature is not None else None
        self.link_feature = link_feature.to(self.device) if link_feature is not None else None
        self.glycan_feature = glycan_feature.to(self.device) if glycan_feature is not None else None
        self._num_unit = len(unit_type) if unit_type is not None else 0
        self._num_link = len(link_type) if link_type is not None else 0
        self._num_glycoword = len(glycoword_type) if glycoword_type is not None else 0
        
    @property
    def num_unit(self) -> int:
        return self._num_unit
    
    @property
    def num_link(self) -> int:
        return self._num_link
        
    @property
    def num_glycoword(self) -> int:
        return self._num_glycoword
    
    @property
    def num_node(self) -> int:
        return self.num_unit
        
    @property
    def num_edge(self) -> int:
        return self.num_link
    
    def to(self, device):
        new_glycan = Glycan(vocab_manager=self.vocab_manager, device=device)
        if self.edge_list is not None:
            new_glycan.edge_list = self.edge_list.to(device)
        if self.unit_type is not None:
            new_glycan.unit_type = self.unit_type.to(device)
        if self.link_type is not None:
            new_glycan.link_type = self.link_type.to(device)
        if self.glycoword_type is not None:
            new_glycan.glycoword_type = self.glycoword_type.to(device)
        if self.unit_feature is not None:
            new_glycan.unit_feature = self.unit_feature.to(device)
        if self.link_feature is not None:
            new_glycan.link_feature = self.link_feature.to(device)
        if self.glycan_feature is not None:
            new_glycan.glycan_feature = self.glycan_feature.to(device)
        new_glycan._num_unit = self._num_unit
        new_glycan._num_link = self._num_link
        new_glycan._num_glycoword = self._num_glycoword
        return new_glycan


class PackedGlycan:
    def __init__(self, glycans: List[Glycan]):
        self.glycans = glycans
        self.batch_size = len(glycans)
        self.device = glycans[0].device if glycans else torch.device("cpu")
        self.unit_type = torch.cat([g.unit_type for g in glycans if g.unit_type is not None])
        self.glycoword_type = torch.cat([g.glycoword_type for g in glycans if g.glycoword_type is not None])
        self.node2graph = self._create_node2graph()
        self.glycoword2graph = self._create_glycoword2graph()
        self.num_units = torch.tensor([g.num_unit for g in glycans], device=self.device)
        self.num_glycowords = torch.tensor([g.num_glycoword for g in glycans], device=self.device)
        
    def _create_node2graph(self) -> torch.Tensor:
        node2graph = []
        for i, glycan in enumerate(self.glycans):
            node2graph.extend([i] * glycan.num_unit)
        return torch.tensor(node2graph, device=self.device)
        
    def _create_glycoword2graph(self) -> torch.Tensor:
        glycoword2graph = []
        for i, glycan in enumerate(self.glycans):
            glycoword2graph.extend([i] * glycan.num_glycoword)
        return torch.tensor(glycoword2graph, device=self.device)


class GlycanParser:
    def __init__(self, vocab_manager):
        self.vocab_manager = vocab_manager
    
    @staticmethod
    def multireplace(string: str, replace_dict: Dict[str, str]) -> str:
        for k, v in replace_dict.items():
            string = string.replace(k, v)
        return string
    
    def get_unit_id2start_end(self, iupac: str) -> Dict[int, List[int]]:
        units_links = [x for x in self.multireplace(iupac, {"[": "", "]": "", ")": "("}).split("(") if x]
        units = [x for x in units_links
                 if not (x.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", x))]
        id2unit = {i: x for i, x in enumerate(units)}
        id2start_end = {}
        curr_position = 0
        for unit_id, unit in id2unit.items():
            start = iupac[curr_position:].index(unit) + curr_position
            id2start_end[unit_id] = [start, start + len(unit)]
            curr_position = start + len(unit)
        return id2start_end
    
    def locate_brackets(self, iupac: str) -> Tuple[List[int], List[int]]:
        left_indices = []
        right_indices = []
        for i, x in enumerate(iupac):
            if x == "[":
                left_indices.append(i)
            if x == "]" and len(left_indices) > len(right_indices):
                right_indices.append(i)
        return left_indices, right_indices
    
    def on_branch(self, index: int, left_indices: List[int], right_indices: List[int]) -> bool:
        in_intervals = [index >= left_index and index <= right_index
                        for left_index, right_index in zip(left_indices, right_indices)]
        return any(in_intervals)
    
    def get_link_type(self, iupac: str) -> Optional[int]:
        left_indices, right_indices = self.locate_brackets(iupac)
        if len(left_indices) == 0 and len(right_indices) == 0:
            link = "".join([x for x in iupac if x not in ["(", ")", "[", "]"]])
            link_core = self._get_core(link)
            link_type = self.vocab_manager.link2id.get(link_core, None)
        elif len(left_indices) != len(right_indices):
            link_type = None
        else:
            link = "".join([x for i, x in enumerate(iupac)
                            if x not in ["(", ")", "[", "]"] and not self.on_branch(i, left_indices, right_indices)])
            link_core = self._get_core(link)
            link_type = self.vocab_manager.link2id.get(link_core, None)
        return link_type
    
    def get_edge_list(self, iupac: str) -> List[List[int]]:
        parts = [x for x in self.multireplace(iupac, {"}": "{"}).split("{") if x]
        edge_list = []
        num_cum_unit = 0
        
        for part in parts:
            unit_id2start_end = self.get_unit_id2start_end(part)
            num_unit = len(unit_id2start_end)
            
            for src_id in range(num_unit):
                for tgt_id in range(src_id + 1, num_unit):
                    src_end = unit_id2start_end[src_id][1]
                    tgt_start = unit_id2start_end[tgt_id][0]
                    assert tgt_start > src_end
                    inner_part = part[src_end:tgt_start]
                    link_type = self.get_link_type(inner_part)
                    if link_type is not None:
                        src = src_id + num_cum_unit
                        tgt = tgt_id + num_cum_unit
                        edge_list += [[src, tgt, link_type], [tgt, src, link_type]]
            num_cum_unit += num_unit
        
        return edge_list
    
    def locate_glycowords(self, iupac: str, glycowords: List[str]) -> List[List]:
        glycoword_location = []
        curr_position = 0
        for glycoword in glycowords:
            start = iupac[curr_position:].index(glycoword) + curr_position
            end = start + len(glycoword)
            glycoword_location.append([glycoword, start, end])
            curr_position = end
        return glycoword_location
    
    def _get_core(self, token: str) -> str:
        if GLYCOWORK_AVAILABLE:
            return tokenization.get_core(token)
        else:
            return token
    
    def from_iupac(self, iupac: str,
                   unit_feature_names: Optional[List[str]] = None,
                   link_feature_names: Optional[List[str]] = None,
                   device: str = "cpu") -> Glycan:
        unit_feature_names = unit_feature_names or ["default"]
        link_feature_names = link_feature_names or ["default"]
        units_links = [x for x in self.multireplace(iupac,
                                                   {"[": "", "]": "", "{": "", "}": "", ")": "("}).split("(") if x]
        units = [x for x in units_links
                 if not (x.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", x))]
        unit_type = []
        unit_features = []
        for unit in units:
            core_unit = self._get_core(unit)
            unit_id = self.vocab_manager.unit2id.get(core_unit, len(self.vocab_manager.units) - 1)
            unit_type.append(unit_id)
            feature = []
            for name in unit_feature_names:
                feat_func = self.vocab_manager.get_unit_feature_func(name)
                feature.extend(feat_func(unit))
            unit_features.append(feature)
        
        unit_type = torch.tensor(unit_type, device=device)
        unit_feature = torch.tensor(unit_features, dtype=torch.float, device=device) if unit_features[0] else None
        edge_list = self.get_edge_list(iupac)
        link_type = []
        link_features = []
        
        for edge in edge_list:
            src_, tgt_, type_ = edge
            link_type.append(type_)
            feature = []
            for name in link_feature_names:
                feat_func = self.vocab_manager.get_link_feature_func(name)
                link_id = self.vocab_manager.id2link.get(type_, "Unknown")
                feature.extend(feat_func(link_id))
            link_features.append(feature)
        
        link_type = torch.tensor(link_type, device=device) if link_type else torch.empty(0, dtype=torch.long, device=device)
        link_feature = torch.tensor(link_features, dtype=torch.float, device=device) if link_features else None
        glycoword_location = self.locate_glycowords(iupac, units_links)
        glycoword_type = []
        last_start, last_end = 0, 0
        
        for glycoword, start, end in glycoword_location:
            for letter in iupac[last_end:start]:
                if letter in ["[", "]", "{", "}"]:
                    glycoword_type.append(self.vocab_manager.glycoword2id[letter])
            core_glycoword = self._get_core(glycoword)
            glycoword_id = self.vocab_manager.glycoword2id.get(core_glycoword,
                                                              self.vocab_manager.glycoword2id["Unknown_Token"])
            glycoword_type.append(glycoword_id)
            last_start, last_end = start, end
        
        for letter in iupac[last_end:]:
            if letter in ["[", "]", "{", "}"]:
                glycoword_type.append(self.vocab_manager.glycoword2id[letter])
        
        glycoword_type = torch.tensor(glycoword_type, device=device)
        
        return Glycan(
            edge_list=edge_list,
            unit_type=unit_type,
            link_type=link_type,
            glycoword_type=glycoword_type,
            unit_feature=unit_feature,
            link_feature=link_feature,
            vocab_manager=self.vocab_manager,
            device=device
        )
