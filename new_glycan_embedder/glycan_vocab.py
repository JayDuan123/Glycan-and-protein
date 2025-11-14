import os
import re
import pickle as pkl
from typing import List, Dict, Optional
import logging

from .glycan_features import FeatureRegistry

logger = logging.getLogger(__name__)


class VocabularyManager:
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path

        if vocab_path and os.path.exists(vocab_path):
            self._load_vocabularies_from_file(vocab_path)
        else:
            logger.warning("Vocabulary file not found, creating minimal vocabulary")
            self._create_minimal_vocabularies()

        self.feature_registry = FeatureRegistry(self.units, self.links)

        logger.info(f"Vocabulary loaded: {len(self.units)} units, {len(self.links)} links, {len(self.glycowords)} glycowords")

    def _load_vocabularies_from_file(self, vocab_path: str):
        try:
            with open(vocab_path, 'rb') as f:
                entities = pkl.load(f)

            self.units = [entity for entity in entities
                          if not (entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity))]

            self.links = [entity for entity in entities
                          if entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity)]

            self.glycowords = entities + ["[", "]", "{", "}", "Unknown_Token"]

            self.unit2id = {x: i for i, x in enumerate(self.units)}
            self.id2unit = {i: x for x, i in self.unit2id.items()}

            self.link2id = {x: i for i, x in enumerate(self.links)}
            self.id2link = {i: x for x, i in self.link2id.items()}

            self.glycoword2id = {x: i for i, x in enumerate(self.glycowords)}
            self.id2glycoword = {i: x for x, i in self.glycoword2id.items()}

            logger.info(f"Loaded vocabulary from {vocab_path}")

        except Exception as e:
            logger.error(f"Error loading vocabulary from {vocab_path}: {e}")
            logger.info("Falling back to minimal vocabulary")
            self._create_minimal_vocabularies()

    def _create_minimal_vocabularies(self):
        self.units = [
            "Glc", "Gal", "Man", "GlcNAc", "GalNAc", "Fuc", "Xyl",
            "Neu5Ac", "Neu5Gc", "Ara", "Rib", "GlcA", "GalA", "ManA",
            "Kdn", "Sia", "Unknown_Unit"
        ]

        self.links = [
            "a1-2", "a1-3", "a1-4", "a1-6",
            "b1-2", "b1-3", "b1-4", "b1-6",
            "a2-3", "a2-6", "b2-3", "b2-6",
            "Unknown_Link"
        ]

        self.glycowords = self.units + self.links + ["[", "]", "{", "}", "Unknown_Token"]

        self.unit2id = {x: i for i, x in enumerate(self.units)}
        self.id2unit = {i: x for x, i in self.unit2id.items()}

        self.link2id = {x: i for i, x in enumerate(self.links)}
        self.id2link = {i: x for x, i in self.link2id.items()}

        self.glycoword2id = {x: i for i, x in enumerate(self.glycowords)}
        self.id2glycoword = {i: x for x, i in self.glycoword2id.items()}

        logger.info("Created minimal vocabularies")

    def get_unit_feature_func(self, name: str):
        return self.feature_registry.get_unit_feature_func(name)

    def get_link_feature_func(self, name: str):
        return self.feature_registry.get_link_feature_func(name)

    def get_glycan_feature_func(self, name: str):
        return self.feature_registry.get_glycan_feature_func(name)

    def get_vocab_sizes(self) -> Dict[str, int]:
        return {
            'num_units': len(self.units),
            'num_links': len(self.links),
            'num_glycowords': len(self.glycowords)
        }

    def save_vocabulary(self, path: str):
        entities = self.units + self.links
        with open(path, 'wb') as f:
            pkl.dump(entities, f)
        logger.info(f"Saved vocabulary to {path}")

    def get_unit_categories(self) -> List[str]:
        from .glycan_features import UNIT2CATEGORY
        return list(set(UNIT2CATEGORY.values()))

    def lookup_unit(self, unit_id: int) -> str:
        return self.id2unit.get(unit_id, "Unknown_Unit")

    def lookup_link(self, link_id: int) -> str:
        return self.id2link.get(link_id, "Unknown_Link")

    def lookup_glycoword(self, glycoword_id: int) -> str:
        return self.id2glycoword.get(glycoword_id, "Unknown_Token")
